from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _safe_inv(a: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a)


def decomp_cpu_custom(M: np.ndarray):
    """
    Perform Lu-Chipman decomposition on a Mueller matrix image.

    Expects M in shape (4, 4, H, W) or (16, H, W).
    Returns: Mdelt, Mr, Md, Delta, R, delta, psi, theta, D, V_CP, V_LP.
    """
    if M.ndim == 3 and M.shape[0] == 16:
        h, w = M.shape[1], M.shape[2]
        M = M.reshape(4, 4, h, w)
    if M.ndim != 4 or M.shape[0:2] != (4, 4):
        raise ValueError(f"Unexpected Mueller shape: {M.shape}")

    M = np.moveaxis(np.moveaxis(M, 0, -1), 0, -1)

    eye = np.eye(4)
    pad_mask = np.all(np.isclose(M, eye, atol=1e-6), axis=(2, 3))

    Nx, Ny, _, _ = M.shape
    Dvec = np.conj(np.transpose(M[:, :, 0, 1:].reshape([Nx, Ny, 1, -1]), (0, 1, 3, 2)))
    Dvec = Dvec.astype(np.complex64)
    D = np.linalg.norm(Dvec, axis=2)
    D = np.reshape(D, (Nx, Ny, 1, 1))
    mD = np.zeros((Nx, Ny, 3, 3))
    eye1133 = np.reshape(np.eye(3), (1, 1, 3, 3))

    ZeroDiatt = np.argwhere(D[..., 0, 0] == 0)
    if ZeroDiatt.size != 0:
        mD[ZeroDiatt[:, 0], ZeroDiatt[:, 1]] = np.eye(3)
    else:
        mD = np.sqrt(1 - D**2) * eye1133 + ((1 - np.sqrt(1 - D**2)) / D**2) * np.matmul(
            Dvec, np.conj(np.transpose(Dvec, (0, 1, 3, 2)))
        )

    D = np.reshape(D, (Nx, Ny))

    ones2D = np.ones((Nx, Ny, 1, 1))
    Md = np.concatenate(
        [
            np.concatenate([ones2D, np.conj(np.transpose(Dvec, (0, 1, 3, 2)))], axis=-1),
            np.concatenate([Dvec, mD], axis=-1),
        ],
        axis=2,
    )

    del ones2D

    Md_safe = Md.copy()
    M_safe = M.copy()

    Md_safe[pad_mask] = np.eye(4)
    M_safe[pad_mask] = np.eye(4)

    MdeltMr = np.matmul(M_safe.astype("complex"), _safe_inv(Md_safe))

    mdash = MdeltMr[:, :, 1:, 1:]

    nanmask = np.isnan(MdeltMr[:, :, 0, 0])
    nanmask = np.reshape(nanmask, (Nx, Ny, 1))
    mdash_real = np.real(mdash)
    mdash_imag = np.imag(mdash)
    mdash_nonan_real = np.nan_to_num(mdash_real)
    mdash_nonan_imag = np.nan_to_num(mdash_imag)

    mdash_nonan = mdash_nonan_real + 1j * mdash_nonan_imag

    del mdash_real, mdash_imag, mdash_nonan_real, mdash_nonan_imag

    v_nonan = np.linalg.eig(mdash_nonan * np.conj(np.transpose(mdash_nonan, (0, 1, 3, 2))))[0]
    v = v_nonan
    v[np.reshape(nanmask, (Nx, Ny))] *= np.nan
    d = np.linalg.det(mdash)

    V_CP = 1 - np.sqrt(v[:, :, 0])
    V_LP = 0.5 * (1 - np.sqrt(v[:, :, 1])) + 1 - np.sqrt(v[:, :, 2])

    signo = d / np.abs(d)
    signo = np.reshape(signo, (Nx, Ny, 1, 1))
    eig_m1 = (
        np.sqrt(v[:, :, 0] * v[:, :, 1])
        + np.sqrt(v[:, :, 1] * v[:, :, 2])
        + np.sqrt(v[:, :, 2] * v[:, :, 0])
    )
    eig_m1 = np.reshape(eig_m1, (Nx, Ny, 1, 1))
    eig_m2 = np.sqrt(v[:, :, 0]) + np.sqrt(v[:, :, 1]) + np.sqrt(v[:, :, 2])
    eig_m2 = np.reshape(eig_m2, (Nx, Ny, 1, 1))
    eig_m3 = np.sqrt(v[:, :, 0] * v[:, :, 1] * v[:, :, 2])
    eig_m3 = np.reshape(eig_m3, (Nx, Ny, 1, 1))

    mdelt = (
        signo
        * _safe_inv(np.matmul(mdash, np.conj(np.transpose(mdash, (0, 1, 3, 2)))) + eig_m1 * eye1133)
        * (eig_m2 * np.matmul(mdash, np.conj(np.transpose(mdash, (0, 1, 3, 2)))) + eig_m3)
        * eye1133
    )

    del signo, eig_m1, eig_m2, eig_m3

    trace = (np.diagonal(mdelt, offset=0, axis1=2, axis2=3)).sum(-1)
    Delta = 1 - np.abs(trace) / 3

    del trace

    Pdelt = MdeltMr[:, :, 1:, 0]
    Pdelt = np.reshape(Pdelt, (Nx, Ny, -1, 1))

    onesvec = np.ones((Nx, Ny, 1, 4)) * np.reshape(np.array((1, 0, 0, 0)), (1, 1, 1, 4))
    Mdelt = np.concatenate([onesvec, np.concatenate([Pdelt, mdelt], axis=3)], axis=2)

    mr = np.matmul(_safe_inv(mdelt), mdash)

    zerosvec = np.zeros((Nx, Ny, 3, 1))
    Mr = np.concatenate([onesvec, np.concatenate([zerosvec, mr], axis=3)], axis=2)

    trace = (np.diagonal(Mr, offset=0, axis1=2, axis2=3)).sum(-1)
    a = trace / 2 - 1
    R = np.arccos(a)
    cond1 = 1 >= np.real(a)
    cond2 = np.real(a) >= -1
    nanmask = cond1 * cond2
    del cond1, cond2

    delta = np.arccos(Mr[:, :, 3, 3])
    psi = np.arccos(np.cos(R / 2) / (delta / 2))

    theta = -np.arctan2((Mr.real[:, :, 1, 3]), (Mr.real[:, :, 2, 3])) / 2
    theta[theta < 0] += np.pi

    Delta[pad_mask] = np.nan
    R[pad_mask] = np.nan
    psi[pad_mask] = np.nan
    theta[pad_mask] = np.nan
    D[pad_mask] = np.nan
    V_CP[pad_mask] = np.nan
    V_LP[pad_mask] = np.nan
    Mdelt[pad_mask] = np.nan
    Mr[pad_mask] = np.nan
    Md[pad_mask] = np.nan

    return Mdelt, Mr, Md, Delta, R, delta, psi, theta, D, V_CP, V_LP


def compute_decomp_maps(mu: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute Lu-Chipman target maps for a single Mueller image.
    Returns (targets, roi_mask) where roi_mask marks valid pixels.
    """
    _, _, _, Delta, R, _, _, theta, D, _, _ = decomp_cpu_custom(mu)
    targets = {
        "retardance": np.clip(np.nan_to_num(R), 0.0, np.pi).astype(np.float32),
        "diattenuation": np.clip(np.nan_to_num(D), 0.0, 1.0).astype(np.float32),
        "depolarization": np.clip(np.nan_to_num(Delta), 0.0, 1.0).astype(np.float32),
    }
    valid = np.isfinite(R) & np.isfinite(D) & np.isfinite(Delta) & np.isfinite(theta)
    roi_mask = valid.astype(np.float32)
    return targets, roi_mask
