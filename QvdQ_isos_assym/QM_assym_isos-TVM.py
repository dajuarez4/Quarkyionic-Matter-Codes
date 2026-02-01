def kfp_solver_CS_TVM(rhob, fq, y, kbu, model="tvm"):
    n_p = rhob*(1.0 - fq)*y
    n_n = rhob*(1.0 - fq)*(1.0 - y)
    x_p = bn*n_p + bpn*n_n

    f = f_tvm(x_p) if model=="tvm" else f_cs(x_p)
    if not np.isfinite(f) or f <= 0:
        return np.nan

    n_p_id = n_p / f
    val = kbu**3 + (6.0*np.pi**2/g)*n_p_id
    return np.cbrt(max(val, 0.0))

def kfn_solver_CS_TVM(rhob, fq, y, kbu, model="tvm"):
    n_p = rhob*(1.0 - fq)*y
    n_n = rhob*(1.0 - fq)*(1.0 - y)
    x_n = bpn*n_p + bn*n_n

    f = f_tvm(x_n) if model=="tvm" else f_cs(x_n)
    if not np.isfinite(f) or f <= 0:
        return np.nan

    n_n_id = n_n / f
    val = kbu**3 + (6.0*np.pi**2/g)*n_n_id
    return np.cbrt(max(val, 0.0))

