import numpy as np

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
)
from meshmode.dof_array import DOFArray

from pytential import bind, sym
from pytential.target import PointsTarget

# my file
from pytential_handling.my_laplace_kernel import ScreenedLaplaceKernel
import sys
from os import devnull


# Disable
def blockPrint():
    sys.stdout = open(devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# {{{ set some constants for use below

nelements = 20
bdry_quad_order = 5 # order of quadrature on the boundary
mesh_order = 5
qbx_order = 5
bdry_ovsmp_quad_order = 4*bdry_quad_order # boundary ? quadrature order
fmm_order = 10
k = 0.25

# }}}

from meshmode.mesh.generation import ellipse, make_curve_mesh
from functools import partial
from meshmode.mesh.processing import affine_map, merge_disjoint_meshes


class AmphilicsSolver:
    def __init__(self, particle_pos, particle_facing, k, field_extent = 30, cogs=1, gamma=4.3):
        from meshmode.array_context import PyOpenCLArrayContext
        import pyopencl as cl

        cl_ctx = cl.create_some_context()
        queue = cl.CommandQueue(cl_ctx)
        allocator = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(queue))

        self.actx = PyOpenCLArrayContext(queue, allocator=allocator)

        self.cogs = cogs
        self.pos_array = particle_pos
        self.facing_array = particle_facing
        self.k=k
        self.gamma = gamma

        self.base_mesh = make_curve_mesh(
            partial(ellipse, 1),
            np.linspace(0, 1, nelements + 1),
            mesh_order
        )

        from sumpy.visualization import FieldPlotter
        fplot = FieldPlotter(np.zeros(2), extent=field_extent, npoints=500)
        self.targets = self.actx.from_numpy(fplot.points)

        # set up symbolics
        from sumpy.kernel import YukawaKernel
        self.kernel = YukawaKernel(2)

        self.sigma_sym = sym.var("sigma")
        self.sqrt_w = sym.sqrt_jac_q_weight(2)
        self.k_sym = sym.var("k")

        self.inv_sqrt_w_sigma = sym.cse(self.sigma_sym / self.sqrt_w)

        loc_sign = -1  # exterior condition DO NOT CHANGE
        self.bdry_op_sym = (-loc_sign * 0.5 * self.sigma_sym
                       + self.sqrt_w * (sym.S(self.kernel, self.inv_sqrt_w_sigma, lam=self.k_sym, qbx_forced_limit=+1)
                                        + sym.D(self.kernel, self.inv_sqrt_w_sigma, lam=self.k_sym,
                                                qbx_forced_limit="avg")))  # }}}

        self.repr_kwargs = {
            "source": "qbx_high_target_assoc_tol",
            "target": "targets",
            "qbx_forced_limit": None
        }
        self.representation_sym = (
                sym.S(self.kernel, self.inv_sqrt_w_sigma, lam=self.k_sym, **self.repr_kwargs)
                + sym.D(self.kernel, self.inv_sqrt_w_sigma, lam=self.k_sym, **self.repr_kwargs)
        )


        # --- gradient ---
        from pytential.symbolic.primitives import grad
        self.grad_sym = grad(ambient_dim=2, operand=self.representation_sym)
        self.nvec_sym = sym.make_sym_vector("normal", 2)

        self.representation_sym_grad = grad(ambient_dim=2, operand=self.representation_sym)

        self.last_gmres = None

        # FOR COMPUTATION OF CFL CONDITION

    def update_particles(self, particle_pos, particle_facing):

        self.pos_array = particle_pos
        self.facing_array = particle_facing

        particle_size = 1.25

        meshes = [
            affine_map(self.base_mesh, A=particle_size*np.diag([1, 1]), b=pos)
            for pos in particle_pos
        ]

        mesh = merge_disjoint_meshes(meshes, single_group=False)

        pre_density_discr = Discretization(self.actx, mesh,
                                           InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))


        from pytential.qbx import QBXLayerPotentialSource
        self.qbx = QBXLayerPotentialSource(
            pre_density_discr,
            fine_order=bdry_ovsmp_quad_order,
            qbx_order=qbx_order,
            fmm_order=fmm_order
        )

        from pytential import GeometryCollection
        self.places = GeometryCollection({
            "qbx": self.qbx,
            "qbx_high_target_assoc_tol":
                self.qbx.copy(target_association_tolerance=0.05),
            "targets": PointsTarget(self.targets)
        }, auto_where="qbx")

        self.density_discr = self.places.get_discretization("qbx")

        from sumpy.kernel import LaplaceKernel

        # --- indicator ---
        self.indicator_op = bind(
            self.places,
            sym.D(LaplaceKernel(2), self.sigma_sym, **self.repr_kwargs)
        )

        # --- normals / weights ---
        self.normal = bind(self.density_discr, sym.normal(2))(self.actx).as_vector(object)

        from pytential.symbolic.primitives import area_element, QWeight
        dS = area_element(1, 1, None) * QWeight(None)
        self.integral_weights = bind(self.density_discr, dS)(self.actx)

        def hydrophobic_stress_T(u_sym, grad_u_sym, gamma=self.gamma, rho=1/self.k):
            # grad_u_sym is expected to be a symbolic vector (e.g., a tuple of expressions)
            grad_x_sym = grad_u_sym[0]
            grad_y_sym = grad_u_sym[1]

            # Magnitude squared of gradient
            grad_mag_sq = grad_x_sym ** 2 + grad_y_sym ** 2

            # Scalar part of the first two terms in the definition of T_ij
            # (u^2/rho) * delta_ij + (1/2) * |grad u|^2 * delta_ij
            scalar_diagonal_term = (u_sym ** 2 / rho) + rho * grad_mag_sq / 2

            # Factor for the outer product term: -2 * rho * (grad_i u) * (grad_j u)
            outer_product_factor = 2 * rho

            T_xx_sym = gamma * (scalar_diagonal_term - outer_product_factor * grad_x_sym * grad_x_sym)
            T_xy_sym = - gamma * (outer_product_factor * grad_x_sym * grad_y_sym)
            T_yx_sym = T_xy_sym
            T_yy_sym = gamma * (scalar_diagonal_term - outer_product_factor * grad_y_sym * grad_y_sym)

            # Return components of the stress tensor as a tuple
            return (T_xx_sym, T_xy_sym, T_yx_sym, T_yy_sym)

        # Use separate symbolic variables for position components for evaluation compatibility
        r_pos_sym = sym.make_sym_vector("r_pos", 2)

        repr_kwargs_boundary = {
            "source": "qbx",  # Source is 'qbx' (pre_density_discr)
            "target": "qbx",  # Target is 'qbx' (the boundary itself)
            "qbx_forced_limit": +1  # Or appropriate limit for boundary evaluation
        }
        representation_sym_boundary = (
                sym.S(self.kernel, self.inv_sqrt_w_sigma, lam=self.k_sym, **repr_kwargs_boundary)
                + sym.D(self.kernel, self.inv_sqrt_w_sigma, lam=self.k_sym, **repr_kwargs_boundary)
        )

        # find grad of potential on the boundary
        from pytential.symbolic.primitives import grad
        representation_sym_grad_boundary = grad(ambient_dim=2, operand=representation_sym_boundary)

        # calculate hydrophobic stress tensor on the boundary
        self.T_sym = hydrophobic_stress_T(representation_sym_boundary, representation_sym_grad_boundary,
                                                         rho=1 / self.k)

        self.T_sym_components = hydrophobic_stress_T(self.representation_sym, self.representation_sym_grad, rho=1 / k)

        # Define force integrands
        self.force_integrand_x_sym = self.T_sym[0] * self.nvec_sym[0] + self.T_sym[1] * self.nvec_sym[1]
        self.force_integrand_y_sym = self.T_sym[1] * self.nvec_sym[0] + self.T_sym[2] * self.nvec_sym[1]

        # formula derived from definitions by me <3
        self.torque_integrand_sym = r_pos_sym[0] * self.force_integrand_y_sym - r_pos_sym[1] * self.force_integrand_x_sym

        # --- nodes cached ---
        self.nodes = self.actx.thaw(self.density_discr.nodes())

        # gradient
        self.grad_op = bind(self.places, self.grad_sym)

        self.T_xx_op = bind(self.places, self.T_sym[0])
        self.T_xy_op = bind(self.places, self.T_sym[1])
        self.T_yy_op = bind(self.places, self.T_sym[2])

        # TORQUE
        self.mv_pos = bind(self.density_discr, sym.nodes(2))(self.actx)
        self.pos = self.mv_pos.as_vector(object)

    def _amphilic_bc(self):

        x, y = self.nodes
        actx = self.actx

        bc_data = []

        for igrp, facing in enumerate(self.facing_array):

            cos_f = actx.np.cos(facing)
            sin_f = actx.np.sin(facing)

            xg = x[igrp] - self.pos_array[igrp,0]
            yg = y[igrp] - self.pos_array[igrp,1]

            rot_x = cos_f * xg + sin_f * yg
            rot_y = -sin_f * xg + cos_f * yg

            theta = actx.np.arctan2(rot_y, rot_x)
            bc = (actx.np.cos(self.cogs*theta)+1)/2

            bc_data.append(bc)

        return DOFArray(actx, tuple(bc_data))

    def solve(self):
        actx = self.actx

        bc = self._amphilic_bc()

        self.bvp_rhs = bind(self.places, self.sqrt_w*sym.var("bc"))(
            actx, bc=bc)
        self.bound_op = bind(self.places, self.bdry_op_sym)

        from pytential.linalg.gmres import gmres

        try:
            gmres_result = gmres(
                self.bound_op.scipy_op(
                    actx, self.sigma_sym.name,
                    dtype=np.complex128, k=self.k),
                self.bvp_rhs,
                tol=1e-8,
                x0 = self.last_gmres
            )
        except:
            gmres_result = gmres(
                self.bound_op.scipy_op(
                    actx, self.sigma_sym.name,
                    dtype=np.complex128, k=self.k),
                self.bvp_rhs,
                tol=1e-8
            )

        self.last_gmres = gmres_result.solution

        return self.compute_hydro_out(self.last_gmres)

    def compute_hydro_out(self, solution):
        actx = self.actx

        # --- field ---
        self.fld = actx.to_numpy(
            bind(self.places, self.representation_sym)(
                actx, sigma=solution, k=self.k)
        ).astype(np.float64)

        # --- indicator ---
        ones_density = self.density_discr.zeros(actx)
        for elem in ones_density:
            elem.fill(1)

        indicator = actx.to_numpy(
            self.indicator_op(actx, sigma=ones_density)
        )

        self.force_density_x = bind(self.places, self.force_integrand_x_sym)(actx, sigma=solution, k=self.k, normal=self.normal)
        self.force_density_y = bind(self.places, self.force_integrand_y_sym)(actx, sigma=solution, k=self.k, normal=self.normal)
        self.torque_density = bind(self.places, self.torque_integrand_sym)(actx, sigma=solution, k=self.k, normal=self.normal,
                                                            r_pos=self.pos)

        n_particles = len(self.force_density_x)

        forces_x = np.ones(n_particles)
        forces_y = np.ones(n_particles)
        torques = np.ones(n_particles)

        for igrp in range(n_particles):
            # manual node.sum
            fx = actx.to_numpy(
                actx.np.sum(self.force_density_x[igrp] * self.integral_weights[igrp])
            )
            fy = actx.to_numpy(
                actx.np.sum(self.force_density_y[igrp] * self.integral_weights[igrp])
            )
            t = actx.to_numpy(
                actx.np.sum(self.torque_density[igrp] * self.integral_weights[igrp])
            )

            forces_x[igrp] = fx
            forces_y[igrp] = fy

            # torque needs to be centred around particle
            torques[igrp] = t - (self.pos_array[igrp][0] * fy - self.pos_array[igrp][1] * fx)

        # --- gradient ---
        grad = self.grad_op(actx, sigma=solution, k=self.k)
        grad_x = actx.to_numpy(grad[0])
        grad_y = actx.to_numpy(grad[1])

        # --- stress tensor ---
        self.T_xx = actx.to_numpy(
            bind(self.places, self.T_sym_components[0])(actx, sigma=solution, k=self.k))
        self.T_xy = actx.to_numpy(
            bind(self.places, self.T_sym_components[1])(actx, sigma=solution, k=self.k))
        self.T_yy = actx.to_numpy(
            bind(self.places, self.T_sym_components[3])(actx, sigma=solution, k=self.k))

        hydro_out = np.array([
            self.fld.flatten(),
            indicator.flatten(),
            grad_x.flatten(),
            grad_y.flatten(),
            self.T_xx.flatten(),
            self.T_yy.flatten(),
            self.T_xy.flatten()
        ], dtype=np.float64)

        self.qbx.qbx_fmm_geometry_data.clear_cache(self.qbx)

        return (forces_x, forces_y, torques), hydro_out

