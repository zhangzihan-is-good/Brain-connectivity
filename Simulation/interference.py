"""The single shared-measurement-interference scenario displayed in panel d.

Susceptible edge probability is .15; .8 is the shared fraction of measurement
noise variance, not the fraction of total BOLD variance. Separate nuisance
random draws leave the original latent-generator stream and task labels intact.
"""
import numpy as np
import core

SETTING = dict(id='shared_q_0.15_f_0.8', susceptible_fraction=.15,
               shared_noise_fraction=.8, noise_ratio=1., length_ratio=1.,
               strength_distribution='original')
INTERFERENCE_STREAM = 2026100701


def generate(cell, split, repeat, variant=0, n_pairs=core.N_PAIRS):
    q, sharing, noise_ratio = .15, .8, 1.
    cfg = core.FIELDS[cell["field"]]
    rng = core.rng_for(cell, split, repeat, variant)
    tr, n = cfg["tr"], cfg["n"]
    pad = 128
    total = n + 2 * pad
    nr = core.N_REFERENCE + core.N_TARGET
    shape = (nr, n_pairs, total)
    kappa = rng.uniform(0.35, 0.8, n_pairs)
    template_d = rng.choice([-2.0, -1.0, 1.0, 2.0], n_pairs)
    template_s = rng.choice([-1.0, 1.0], n_pairs)
    tsnr = np.clip(rng.normal(cfg["tsnr"], cfg["tsnr_sd"], n_pairs),
                   cfg["tsnr"] - 2 * cfg["tsnr_sd"], cfg["tsnr"] + 2 * cfg["tsnr_sd"])
    d = np.broadcast_to(template_d, (nr, n_pairs)).copy()
    signs = np.broadcast_to(template_s, (nr, n_pairs)).copy()
    present = np.ones((nr, n_pairs), dtype=bool)
    relation = cell["relation"]
    if relation == "sync":
        template_d[:] = 0
        d[:] = 0
    if relation == "lag_variable":
        d = rng.choice([-2.0, -1.0, 1.0, 2.0], (nr, n_pairs))
    if relation == "sign_variable":
        signs = rng.choice([-1.0, 1.0], (nr, n_pairs))
    if relation == "sporadic":
        present = rng.random((nr, n_pairs)) < 0.3
    if relation == "independent":
        present[:] = False
    cost = np.abs(d)
    if cell["family"] == "attenuation":
        attenuation = np.exp(-0.18 * cost)
        width = np.zeros_like(cost)
        jitter = rng.normal(0, 0.08 * cost)
    elif cell["family"] == "dispersion":
        attenuation = np.exp(-0.06 * cost)
        width = 0.3 * cost
        jitter = rng.normal(0, 0.12 * cost)
    else:
        attenuation = np.ones_like(cost)
        width = np.zeros_like(cost)
        jitter = np.zeros_like(cost)
    actual_d = d + jitter
    u, f = core.spectrum_process(rng, shape, tr, True, core.KNEE)
    v, _ = core.spectrum_process(rng, shape, tr, True, core.KNEE)
    ex, _ = core.spectrum_process(rng, shape, tr, False, core.KNEE)
    ey, _ = core.spectrum_process(rng, shape, tr, False, core.KNEE)
    phase = np.exp(-2j * np.pi * actual_d[..., None] * f)
    kernel = np.exp(-2 * np.pi ** 2 * width[..., None] ** 2 * f ** 2)
    driven = np.fft.irfft(np.fft.rfft(u, axis=-1) * phase * kernel, n=total, axis=-1)
    driven_variance = np.mean(driven ** 2, axis=-1)
    a = kappa[None, :] * attenuation * present
    latent_y = signs[..., None] * a[..., None] * driven + np.sqrt(
        np.maximum(1 - a ** 2 * driven_variance, 0))[..., None] * v

    susceptible = np.zeros(n_pairs, dtype=bool)
    interference_d = np.zeros((nr, n_pairs))
    interference_s = np.zeros((nr, n_pairs))
    if q > 0 and sharing > 0:
        interference_rng = np.random.default_rng(np.random.SeedSequence(
            [core.BASE_SEED, cell["id"], split, repeat, variant, INTERFERENCE_STREAM]))
        susceptible = interference_rng.random(n_pairs) < q
        interference_d = interference_rng.choice([-2.0, -1.0, 1.0, 2.0], (nr, n_pairs))
        interference_s = interference_rng.choice([-1.0, 1.0], (nr, n_pairs))
        nuisance_phase = np.exp(-2j * np.pi * interference_d[..., None] * f)
        shifted_noise = np.fft.irfft(np.fft.rfft(ex, axis=-1) * nuisance_phase,
                                     n=total, axis=-1)
        mixture = np.sqrt(sharing) * interference_s[..., None] * shifted_noise + np.sqrt(1 - sharing) * ey
        # Keep each measurement-noise series at the original unit variance.
        mixture = core.unit(mixture)
        ey = np.where(susceptible[None, :, None], mixture, ey)

    sx = (u + np.sqrt(noise_ratio) * ex) / np.sqrt(1 + noise_ratio)
    sy = (latent_y + np.sqrt(noise_ratio) * ey) / np.sqrt(1 + noise_ratio)
    sx, sy = core.unit(sx[..., pad:pad + n]), core.unit(sy[..., pad:pad + n])
    scale = 100 / tsnr
    x = (100 + sx * scale[None, :, None]).astype(np.float32)
    y = (100 + sy * scale[None, :, None]).astype(np.float32)
    return dict(x=x, y=y, kappa=kappa, template_d=template_d, template_s=template_s,
                actual_d=actual_d, nominal_d=d, actual_s=signs, present=present,
                tsnr=tsnr, noise_ratio=np.array(noise_ratio), knee=np.array(core.KNEE),
                cell_id=np.array(cell["id"]), split=np.array(split), repeat=np.array(repeat),
                nuisance_susceptible=susceptible, nuisance_lag=interference_d,
                nuisance_sign=interference_s)


def generate_interference(cell, split, repeat, purpose):
    # The exploratory setting and its null thresholds were frozen before the
    # independent validation streams used for the displayed test results.
    variant = (202610060101 if split == 0 else 202610060201) if purpose == "direction" else 202610060202
    return generate(cell, split, repeat, variant=variant)
