# Copyright 2023 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

from __future__ import annotations

from abc import ABC
import pathlib
from typing import Callable, Optional, Tuple

from h5netcdf.legacyapi import Dataset
import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from scotty.derivatives import derivative
from scotty.fun_general import find_nearest
from scotty.typing import ArrayLike, FloatArray


class MagneticField(ABC):
    """Abstract base class for magnetic field geometries"""

    #: Sample locations for the major radius coordinate
    R_coord: FloatArray
    #: Sample locations for the vertical coordinate
    Z_coord: FloatArray
    #: Value of the poloidal magnetic flux, :math:`\psi`, on ``(R_coord, Z_coord)``
    poloidalFlux_grid: FloatArray
    ## TODO: Include B_R_grid, B_T_grid, and B_Z_grid

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def d_poloidal_flux_dR(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux, "q_R", {"q_R": q_R, "q_Z": q_Z}, {"q_R": delta_R}
        )

    def d_poloidal_flux_dZ(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux, "q_Z", {"q_R": q_R, "q_Z": q_Z}, {"q_Z": delta_Z}
        )

    def d2_poloidal_flux_dR2(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_R", "q_R"),
            {"q_R": q_R, "q_Z": q_Z},
            {"q_R": delta_R},
        )

    def d2_poloidal_flux_dZ2(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_Z", "q_Z"),
            {"q_R": q_R, "q_Z": q_Z},
            {"q_Z": delta_Z},
        )

    def d2_poloidal_flux_dRdZ(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float, delta_Z: float
    ) -> FloatArray:
        return derivative(
            self.poloidal_flux,
            ("q_R", "q_Z"),
            {"q_R": q_R, "q_Z": q_Z},
            {"q_R": delta_R, "q_Z": delta_Z},
        )

    def magnitude(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        """Returns :math:`|B|`, the magnitude of the magnetic field"""
        return np.sqrt(
            self.B_R(q_R, q_Z) ** 2 + self.B_T(q_R, q_Z) ** 2 + self.B_Z(q_R, q_Z) ** 2
        )

    def unit(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}/|B|`, the unit vector of the magnetic field"""
        magnitude = self.magnitude(q_R, q_Z)
        unit_vector = np.array(
            [self.B_R(q_R, q_Z), self.B_T(q_R, q_Z), self.B_Z(q_R, q_Z)]
        )
        return (unit_vector / magnitude).T


class CircularCrossSectionField(MagneticField):
    """Simple circular cross-section magnetic geometry

    Parameters
    ----------
    B_T_axis:
        Toroidal magnetic field at the magnetic axis (Tesla)
    R_axis:
        Major radius of the magnetic axis (metres)
    minor_radius_a:
        Minor radius of the last closed flux surface (metres)
    B_p_a:
        Poloidal magnetic field at ``minor_radius_a`` (Tesla)
    R_points:
    Z_points:
        Number of points for sample ``(R, Z)`` grid
    grid_buffer_factor:
        Multiplicative factor to increase size of sample grid by
    """

    def __init__(
        self,
        B_T_axis: float,
        R_axis: float,
        minor_radius_a: float,
        B_p_a: float,
        R_points: int = 101,
        Z_points: int = 101,
        grid_buffer_factor: float = 1.0,
    ):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        self.B_p_a = B_p_a

        grid_width = grid_buffer_factor * minor_radius_a
        self.R_coord = np.linspace(R_axis - grid_width, R_axis + grid_width, R_points)
        self.Z_coord = np.linspace(-grid_width, grid_width, Z_points)
        self.poloidalFlux_grid = self.poloidal_flux(
            *np.meshgrid(self.R_coord, self.Z_coord, indexing="ij")
        )

    def rho(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.sqrt((q_R - self.R_axis) ** 2 + q_Z**2)

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.where(
            abs(q_Z) < 1e-12,
            0.0,
            self.B_p_a * q_Z / (q_R * self.minor_radius_a * self.rho(q_R, q_Z)),
        )

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * (self.R_axis / q_R)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.where(
            abs(q_R - self.R_axis) < 1e-12,
            0.0,
            -self.B_p_a
            * (q_R - self.R_axis)
            / (q_R * self.minor_radius_a * self.rho(q_R, q_Z)),
        )

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.rho(q_R, q_Z) / self.minor_radius_a


class ConstantCurrentDensityField(MagneticField):
    """Circular cross-section magnetic geometry with constant current density

    TODO
    ----
    Poloidal flux needs to be made consistent with  B_{R,Z}

    Parameters
    ----------
    B_T_axis:
        Toroidal magnetic field at the magnetic axis (Tesla)
    R_axis:
        Major radius of the magnetic axis (metres)
    minor_radius_a:
        Minor radius of the last closed flux surface (metres)
    B_p_a:
        Poloidal magnetic field at ``minor_radius_a`` (Tesla)
    R_points:
    Z_points:
        Number of points for sample ``(R, Z)`` grid
    grid_buffer_factor:
        Multiplicative factor to increase size of sample grid by
    """

    def __init__(
        self,
        B_T_axis: float,
        R_axis: float,
        minor_radius_a: float,
        B_p_a: float,
        R_points: int = 101,
        Z_points: int = 101,
        grid_buffer_factor: float = 1,
    ):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        self.B_p_a = B_p_a

        grid_width = grid_buffer_factor * minor_radius_a
        self.R_coord = np.linspace(R_axis - grid_width, R_axis + grid_width, R_points)
        self.Z_coord = np.linspace(-grid_width, grid_width, Z_points)
        self.poloidalFlux_grid = self.poloidal_flux(
            *np.meshgrid(self.R_coord, self.Z_coord, indexing="ij")
        )

    def rho(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.sqrt((q_R - self.R_axis) ** 2 + q_Z**2)

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_p_a * q_Z / self.rho(q_R, q_Z)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * (self.R_axis / q_R)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return -self.B_p_a * (q_R - self.R_axis) / self.rho(q_R, q_Z)

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.rho(q_R, q_Z) / self.minor_radius_a


class CurvySlabField(MagneticField):
    """Analytical curvy slab geometry"""

    def __init__(self, B_T_axis: float, R_axis: float):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.zeros_like(q_R)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * self.R_axis / q_R

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.zeros_like(q_R)


def _make_rect_spline(
    R_grid, Z_grid, array, interp_order: int, interp_smoothing: int
) -> Tuple[Callable[[ArrayLike, ArrayLike], FloatArray], RectBivariateSpline]:
    spline = RectBivariateSpline(
        R_grid,
        Z_grid,
        array,
        bbox=[None, None, None, None],
        kx=interp_order,
        ky=interp_order,
        s=interp_smoothing,
    )
    return lambda q_R, q_Z: spline(q_R, q_Z, grid=False), spline


def _make_spline_derivatives(
    spline: RectBivariateSpline,
) -> Tuple[
    Callable[[ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, float, float], FloatArray],
]:
    dpsi_dR = spline.partial_derivative(1, 0)
    dpsi_dZ = spline.partial_derivative(0, 1)
    d2psi_dR2 = spline.partial_derivative(2, 0)
    d2psi_dZ2 = spline.partial_derivative(0, 2)
    d2psi_dRdZ = spline.partial_derivative(1, 1)

    return (
        lambda q_R, q_Z, *args: dpsi_dR(q_R, q_Z, grid=False),
        lambda q_R, q_Z, *args: dpsi_dZ(q_R, q_Z, grid=False),
        lambda q_R, q_Z, *args: d2psi_dR2(q_R, q_Z, grid=False),
        lambda q_R, q_Z, *args: d2psi_dZ2(q_R, q_Z, grid=False),
        lambda q_R, q_Z, *args: d2psi_dRdZ(q_R, q_Z, grid=False),
    )


class InterpolatedField(MagneticField):
    """Interpolated numerical equilibrium using bivariate splines

    Parameters
    ----------
    R_grid:
        1D array of points in ``R`` (metres)
    Z_grid:
        1D array of points in ``Z`` (metres)
    B_R:
        2D ``(R, Z)`` grid of radial magnetic field values (Tesla)
    B_T:
        2D ``(R, Z)`` grid of toroidal magnetic field values (Tesla)
    B_Z:
        2D ``(R, Z)`` grid of vertical magnetic field values (Tesla)
    psi:
        2D ``(R, Z)`` grid of poloidal flux values (Weber/radian)
    interp_order:
        Order of interpolating splines
    interp_smoothing:
        Smoothing factor for interpolating splines
    """

    def __init__(
        self,
        R_grid: FloatArray,
        Z_grid: FloatArray,
        B_R: FloatArray,
        B_T: FloatArray,
        B_Z: FloatArray,
        psi: FloatArray,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ):
        self._interp_B_R, _ = _make_rect_spline(
            R_grid, Z_grid, B_R, interp_order, interp_smoothing
        )
        self._interp_B_T, _ = _make_rect_spline(
            R_grid, Z_grid, B_T, interp_order, interp_smoothing
        )
        self._interp_B_Z, _ = _make_rect_spline(
            R_grid, Z_grid, B_Z, interp_order, interp_smoothing
        )
        self._interp_poloidal_flux, psi_spline = _make_rect_spline(
            R_grid, Z_grid, psi, interp_order, interp_smoothing
        )
        self._set_poloidal_flux_derivatives(psi_spline)

        self.R_coord = R_grid
        self.Z_coord = Z_grid
        self.poloidalFlux_grid = psi

    def _set_poloidal_flux_derivatives(self, psi_spline):
        try:
            (
                self._dpsi_dR,
                self._dpsi_dZ,
                self._d2psi_dR2,
                self._d2psi_dZ2,
                self._d2psi_dRdZ,
            ) = _make_spline_derivatives(psi_spline)
        except AttributeError:
            # Older versions of SciPy don't have
            # `RectBivariateSpline.partial_derivative`, so fall back
            # to base class implementation of flux derivatives
            self._dpsi_dR = super().d_poloidal_flux_dR
            self._dpsi_dZ = super().d_poloidal_flux_dZ
            self._d2psi_dR2 = super().d2_poloidal_flux_dR2
            self._d2psi_dZ2 = super().d2_poloidal_flux_dZ2
            self._d2psi_dRdZ = super().d2_poloidal_flux_dRdZ

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_B_R(q_R, q_Z)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_B_T(q_R, q_Z)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_B_Z(q_R, q_Z)

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_poloidal_flux(q_R, q_Z)

    def d_poloidal_flux_dR(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        return self._dpsi_dR(q_R, q_Z, delta_R)

    def d_poloidal_flux_dZ(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        return self._dpsi_dZ(q_R, q_Z, delta_R)

    def d2_poloidal_flux_dR2(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        return self._d2psi_dR2(q_R, q_Z, delta_R)

    def d2_poloidal_flux_dZ2(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        return self._d2psi_dZ2(q_R, q_Z, delta_R)

    def d2_poloidal_flux_dRdZ(
        self, q_R: ArrayLike, q_Z: ArrayLike, delta_R: float, delta_Z: float
    ) -> FloatArray:
        return self._d2psi_dRdZ(q_R, q_Z, delta_R, delta_Z)


class EFITField(InterpolatedField):
    def __init__(
        self,
        R_grid: FloatArray,
        Z_grid: FloatArray,
        rBphi: FloatArray,
        psi_norm_2D: FloatArray,
        psi_unnorm_axis: float,
        psi_unnorm_boundary: float,
        psi_norm_1D: Optional[FloatArray] = None,
        delta_R: Optional[float] = 0.0001,
        delta_Z: Optional[float] = 0.0001,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ):
        self.R_coord = R_grid
        self.Z_coord = Z_grid
        self.poloidalFlux_grid = psi_norm_2D

        self.delta_R = delta_R or 1e-8
        self.delta_Z = delta_Z or 1e-8

        self._interp_poloidal_flux, psi_spline = _make_rect_spline(
            R_grid, Z_grid, psi_norm_2D, interp_order, interp_smoothing
        )
        self._set_poloidal_flux_derivatives(psi_spline)

        self.poloidal_flux_gradient = psi_unnorm_boundary - psi_unnorm_axis
        if psi_norm_1D is None:
            psi_norm_1D = np.linspace(0, 1.0, len(rBphi))

        self._interp_rBphi = UnivariateSpline(
            psi_norm_1D,
            rBphi,
            w=None,
            bbox=[None, None],
            k=interp_order,
            s=interp_smoothing,
            ext=0,
            check_finite=False,
        )

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        dpolflux_dZ = self.d_poloidal_flux_dZ(q_R, q_Z, self.delta_Z)
        return -dpolflux_dZ * self.poloidal_flux_gradient / q_R

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        polflux = self._interp_poloidal_flux(q_R, q_Z)
        return self._interp_rBphi(polflux) / q_R

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        dpolflux_dR = self.d_poloidal_flux_dR(q_R, q_Z, self.delta_R)
        return dpolflux_dR * self.poloidal_flux_gradient / q_R

    @classmethod
    def from_EFITpp(
        cls,
        filename: pathlib.Path,
        equil_time: float,
        delta_R: Optional[float],
        delta_Z: Optional[float],
        interp_order: int,
        interp_smoothing: int,
    ):
        with Dataset(filename) as dataset:
            efitpp_times = dataset.variables["time"][:]
            time_idx = find_nearest(efitpp_times, equil_time)
            print("EFIT++ time", efitpp_times[time_idx])

            output_group = dataset.groups["output"]
            profiles2D = output_group.groups["profiles2D"]
            # unnormalised, as a function of R and Z
            unnorm_psi_2D = profiles2D.variables["poloidalFlux"][time_idx][:][:]
            data_R_coord = profiles2D.variables["r"][time_idx][:]
            data_Z_coord = profiles2D.variables["z"][time_idx][:]

            fluxFunctionProfiles = output_group.groups["fluxFunctionProfiles"]
            norm_psi_1D = fluxFunctionProfiles.variables["normalizedPoloidalFlux"][:]
            # poloidalFlux as a function of normalised poloidal flux
            unorm_psi_1D = fluxFunctionProfiles.variables["poloidalFlux"][time_idx][:]
            rBphi = fluxFunctionProfiles.variables["rBphi"][time_idx][:]

        # linear fit
        polflux_const_m, polflux_const_c = np.polyfit(unorm_psi_1D, norm_psi_1D, 1)
        norm_psi_2D = unnorm_psi_2D * polflux_const_m + polflux_const_c

        return EFITField(
            R_grid=data_R_coord,
            Z_grid=data_Z_coord,
            rBphi=rBphi,
            psi_norm_2D=norm_psi_2D,
            psi_norm_1D=norm_psi_1D,
            psi_unnorm_axis=0.0,
            psi_unnorm_boundary=polflux_const_m,
            delta_R=delta_R,
            delta_Z=delta_Z,
            interp_order=interp_order,
            interp_smoothing=interp_smoothing,
        )

    @classmethod
    def from_MAST_saved(
        cls,
        filename: pathlib.Path,
        equil_time: float,
        delta_R: Optional[float],
        delta_Z: Optional[float],
        interp_order: int,
        interp_smoothing: int,
    ):
        with np.load(filename) as loadfile:
            # On time base C
            rBphi_all_times = loadfile["rBphi"]
            t_base_B = loadfile["t_base_B"]
            t_base_C = loadfile["t_base_C"]
            data_R_coord = loadfile["R_EFIT"]
            data_Z_coord = loadfile["Z_EFIT"]
            # Time base C
            polflux_axis_all_times = loadfile["poloidal_flux_unnormalised_axis"]
            # Time base C
            psi_boundary_all_times = loadfile["poloidal_flux_unnormalised_boundary"]
            # On time base B
            unnorm_psi_all_times = loadfile["poloidal_flux_unnormalised"]

        t_base_B_idx = find_nearest(t_base_B, equil_time)
        # Get the same time slice
        t_base_C_idx = find_nearest(t_base_C, t_base_B[t_base_B_idx])
        print("EFIT time", t_base_B[t_base_B_idx])

        rBphi = rBphi_all_times[t_base_C_idx, :]
        psi_axis = polflux_axis_all_times[t_base_C_idx]
        psi_boundary = psi_boundary_all_times[t_base_C_idx]
        unnorm_psi_2D = unnorm_psi_all_times[t_base_B_idx, :, :].T

        # Taken from an old file of Sam Gibson's. Should probably check with Lucy or Sam.
        poloidalFlux = np.linspace(0, 1.0, len(rBphi))
        poloidalFlux_grid = (unnorm_psi_2D - psi_axis) / (psi_boundary - psi_axis)

        return cls(
            R_grid=data_R_coord,
            Z_grid=data_Z_coord,
            rBphi=rBphi,
            psi_norm_2D=poloidalFlux_grid,
            psi_norm_1D=poloidalFlux,
            psi_unnorm_axis=psi_axis,
            psi_unnorm_boundary=psi_boundary,
            delta_R=delta_R,
            delta_Z=delta_Z,
            interp_order=interp_order,
            interp_smoothing=interp_smoothing,
        )

    @classmethod
    def from_MAST_U_saved(
        cls,
        filename: pathlib.Path,
        equil_time: float,
        delta_R: Optional[float],
        delta_Z: Optional[float],
        interp_order: int,
        interp_smoothing: int,
    ):
        with np.load(filename) as loadfile:
            time_EFIT = loadfile["time_EFIT"]
            t_idx = find_nearest(time_EFIT, equil_time)
            print("EFIT time", time_EFIT[t_idx])

            return EFITField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                rBphi=loadfile["rBphi"][t_idx, :],
                psi_norm_2D=loadfile["poloidalFlux_grid"][t_idx, :, :],
                psi_norm_1D=loadfile["poloidalFlux"][t_idx, :],
                psi_unnorm_axis=loadfile["poloidalFlux_unnormalised_axis"][t_idx],
                psi_unnorm_boundary=loadfile["poloidalFlux_unnormalised_boundary"][
                    t_idx
                ],
                delta_R=delta_R,
                delta_Z=delta_Z,
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )

    @classmethod
    def from_HL3_MDSplus(
        cls,
        shot: float,
        equil_time: float,
        # delta_R: Optional[float],
        # delta_Z: Optional[float],
        interp_order: int,
        interp_smoothing: int,
        
        mds_server: str = "192.168.20.10",
        diag: str = "EFIT_HL3",
    ):
        """Get EFIT data from HL-3 MDSplus database
        
        Parameters
        ----------
        equil_time : float
            Time point for equilibrium data (seconds)
        delta_R : float, optional
            Finite difference step size for R derivatives
        delta_Z : float, optional
            Finite difference step size for Z derivatives
        interp_order : int
            Order of interpolation spline
        interp_smoothing : int
            Smoothing factor for interpolation
        shot : int, optional
            Shot number
        mds_server : str
            MDSplus server address
        diag : str
            EFIT diagnostic name
            
        Returns
        -------
        EFITField
            Interpolated magnetic field object
        """
        try:
            import MDSplus as mds
        except ImportError:
            raise ImportError("MDSplus package is required for HL3_MDSplus")
            
        # 连接到MDSplus服务器
        try:
            connection = mds.Connection(mds_server)
            connection.openTree(diag, shot)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MDSplus server: {e}")
            
        # 从map_equ.py中借鉴的HL-3网格定义
        # HL-3的R和Z网格（129x129）
        R_grid = np.linspace(1.05, 1.05 + 1.46, 129)  # R网格 [m]
        Z_grid = np.linspace(-3.02/2, 3.02/2, 129)    # Z网格 [m]

        delta_R = R_grid[2] - R_grid[1]  # R网格间隔
        delta_Z = Z_grid[2] - Z_grid[1]  # Z网格间隔
        
        # 读取EFIT时间序列
        try:
            # 获取时间序列
            t_eq = np.atleast_1d(connection.get('\\EFIT_HL3::TOP.TIME').data())
            t_idx = find_nearest(t_eq, equil_time)
            print(f"HL-3 EFIT time: {t_eq[t_idx]:.4f} s")
            
            # 获取规范化的极向磁通网格数据
            psi_2D = connection.get('\\EFIT_HL3::TOP.EFIT_PSIRZ').data().T[:, :, t_idx]
            
            # 获取磁轴和分界面处的极向磁通值
            psi_unnorm_axis = connection.get('\\EFIT_HL3::TOP.EFIT_SIMAG').data()[t_idx]
            psi_unnorm_boundary = connection.get('\\EFIT_HL3::TOP.EFIT_SIBRY').data()[t_idx]

            psi_norm_2D = (psi_2D-psi_unnorm_axis) / (psi_unnorm_boundary-psi_unnorm_axis)
            
            # 根据HL-3的MDSplus结构获取rBphi数据
            # 首先获取规范化的极向磁通一维数组
            # psi_norm_1D = np.linspace(0, 1, 129)
            
            # 获取rBphi数据
            fpol_data = connection.get('\\EFIT_HL3::TOP.EFIT_FPOL').data().T[:, t_idx]
            
            # rBphi需要乘以2*pi/mu_0，参考map_equ.py
            from scipy.constants import mu_0
            rBphi = fpol_data * mu_0

            psi_norm_1D = np.linspace(0, 1.0, len(rBphi))
           
        except Exception as e:
            raise ValueError(f"Error retrieving HL-3 EFIT data: {e}")
            
        # 返回EFITField对象
        return EFITField(
            R_grid=R_grid,
            Z_grid=Z_grid,
            rBphi=rBphi,
            psi_norm_2D=psi_norm_2D,
            psi_norm_1D=psi_norm_1D,
            psi_unnorm_axis=psi_unnorm_axis * 2 * np.pi,  # 转换单位为Weber/radian
            psi_unnorm_boundary=psi_unnorm_boundary * 2 * np.pi,  # 转换单位为Weber/radian
            delta_R=delta_R,
            delta_Z=delta_Z,
            interp_order=interp_order,
            interp_smoothing=interp_smoothing,
        )
