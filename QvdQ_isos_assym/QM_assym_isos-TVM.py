import numpy as np
import matplotlib.pyplot as plt# Properties for nucleons and quarks
g =2; mp = mn = 0.938 #GeV/c**2
Nc = 3; mu = md = mp/Nc; lam = 0.2 #GeV200;
rho0 = 1.23e-3  #GeV^3
rhob_list = np.linspace(1e-6, 5.0, 30) * rho0
y = 0.1
fq_list = np.linspace(0.0, 1.0, 30)
print(len(fq_list))


bn = bpn = an = apn = None

isos_repulsion = "eq"
model = 'cs'
def main(isos_repulsion,model):
    global bn, bpn, an, apn

    if model == 'cs':
        print('cs model')
        # -------------------- CS parameters 
        if isos_repulsion == "eq":   
            an  = 36.9
            bn  = 577.
            apn = 1.45 * an   
            bpn = 1.00 * bn   
            print(r'isospin-blind repulsion $b_n = b_{pn}$')
        elif isos_repulsion == "neq":   
            an  = 26.6
            bn  = 383.
            apn = 2.40 * an   
            bpn = 2.01 * bn   
            print(r'isospin-dependent repulsion $b_n \neq b_{pn}$') 
    elif model == 'tvm':
        print('tvm model')
        # -------------------- TVM parameters
        an = 26.4
        bn = 361 

        if isos_repulsion == 'eq':
        # bpn = bn
            apn = 1.43 * an
            bpn = 1.0 * bn
            print(r'isospin-blind repulsion $b_n = b_{pn}$')

        elif isos_repulsion == 'neq':
        # bpn neq bn
            apn = 2.44 * an
            bpn = 2.08 * bn
            print(r'isospin-dependent repulsion $b_n \neq b_{pn}$') 


    # ------------- help-functions 
    def integrate_simpson(f, a, b, n=1000):
        h = (b - a) / n
        s = f(a) + f(b)
        x = a
        for i in range(1, n):
            x += h
            s += (4.0 if (i % 2) == 1 else 2.0) * f(x)
        return s * (h / 3.0)

    # Ideal quarks u,d energy density

    def f1(q):
        return q * np.sqrt(lam**2 + q**2) * np.sqrt(mu**2 + q**2)
    def f2(q):
        return q * np.sqrt(lam**2 + q**2) * np.sqrt(md**2 + q**2)

    def euid(kbu, y, N=1000):
        k = g / (2 * np.pi**2)
        result = (1+y)/(2-y) * integrate_simpson(f1,0,kbu / Nc)#simpson(f, x=q)
        return Nc * k * result
    def edid(kbu, N=1000):
        k = g / (2 * np.pi**2)
        result = integrate_simpson(f2,0,kbu / Nc)
        return Nc * k * result

    # Ideal proton, neutrons energy density

    def f3(k):
        return k**2 * np.sqrt(mn**2 + k**2)
    def f4(k):
        return k**2 * np.sqrt(mp**2 + k**2)
        
    def enid(kbu, kfn, N=1000):
        k = g / (2 * np.pi**2)
        result = integrate_simpson(f3,kbu, kfn)
        return k * result
    def epid(kbu, kfp, N=1000): 
        k = g / (2 * np.pi**2)
        result = integrate_simpson(f4,kbu, kfp)
        return k * result

    # Ideal quarks u,d baryon density

    def f5(q):
        return q * np.sqrt(lam**2 + q**2) 
    def f6(q):
        return q * np.sqrt(lam**2 + q**2) 

    def nuid(kbu, y, N=1000):
        k = g / (2 * np.pi**2)
        result = (1+y)/(2-y) * integrate_simpson(f5,0,kbu / Nc)#simpson(f, x=q)
        return Nc * k * result
    def ndid(kbu, N=1000):
        k = g / (2 * np.pi**2)
        result = integrate_simpson(f6,0,kbu / Nc)
        return Nc * k * result
        
    # Ideal proton, neutrons baryon density

    def f7(k):
        return k**2 
    def f8(k):
        return k**2 
        
    def nnid(kbu, kfn, N=1000):
        k = g / (2 * np.pi**2)
        result = integrate_simpson(f7,kbu, kfn)
        return k * result
    def npid(kbu, kfp, N=1000): 
        k = g / (2 * np.pi**2)
        result = integrate_simpson(f7,kbu, kfp)
        return k * result

    def kbu_solver(rhob, fq, y):
        k = g / (2.0*np.pi**2)
        w = (3.0/(2.0 - y)) * k
        nq = rhob * fq
        inside = (3.0*nq/w + lam**3)**(2.0/3.0) - lam**2
        return Nc * np.sqrt(max(inside, 0.0))


    def f_tvm(x):
        return np.exp(-x - 0.5*x**2)

    def f_cs(x):
        out = np.full_like(x, np.nan, dtype=float)
        m = x < 4.0
        xm = x[m]
        out[m] = np.exp(-(3*xm)/(4 - xm) - (4*xm)/((4 - xm)**2))
        return out

    def kfp_solver_CS_TVM(rhob, fq, y, kbu, model="tvm"):
        global bpn, bn
        n_p = rhob*(1.0 - fq)*y
        n_n = rhob*(1.0 - fq)*(1.0 - y)
        x_p = (bn*n_p + bpn*n_n)

        f = f_tvm(x_p) if model=="tvm" else f_cs(x_p)
        if not np.isfinite(f) or f <= 0:
            return np.nan

        n_p_id = n_p / f
        val = kbu**3 + (6.0*np.pi**2/g)*n_p_id
        return np.cbrt(max(val, 0.0))

    def kfn_solver_CS_TVM(rhob, fq, y, kbu, model="tvm"):
        global bpn, bn
        n_p = rhob*(1.0 - fq)*y
        n_n = rhob*(1.0 - fq)*(1.0 - y)
        x_n = bpn*n_p + bn*n_n

        f = f_tvm(x_n) if model=="tvm" else f_cs(x_n)
        if not np.isfinite(f) or f <= 0:
            return np.nan

        n_n_id = n_n / f
        val = kbu**3 + (6.0*np.pi**2/g)*n_n_id
        return np.cbrt(max(val, 0.0))

    # colors = ['black','red','yellow','green','blue']

    colors = ['blue', 'deepskyblue', 'cyan', 'springgreen', 'magenta', 'purple']

    y_list = [0.0,0.1, 0.2, 0.3, 0.4, 0.5]

    plt.figure()

    for iy, y in enumerate(y_list):
        energy_density_list = []   

        for rhob in rhob_list:
            xx = []

            for fq in fq_list:
                
                kbu = kbu_solver(rhob, fq, y)
                kfp = kfp_solver_CS_TVM(rhob, fq, y, kbu,model = model)
                kfn = kfn_solver_CS_TVM(rhob, fq, y, kbu,model=model)

                eu = euid(kbu, y=y)
                ed = edid(kbu)
                en = enid(kbu, kfn)
                ep = epid(kbu, kfp)
                
                n_p = rhob * (1.0 - fq) * y
                n_n = rhob * (1.0 - fq) * (1.0 - y)

                x_p = (bn*n_p + bpn*n_n)
                x_n = (bpn*n_p + bn*n_n)
                
                # if (x_p >= 1/4) or (x_n >= 1/4):
                #     xx.append(np.nan)
                #     continue
                if model == "cs" and ((x_p >= 4.0) or (x_n >= 4.0)):
                    xx.append(np.nan)
                    continue
                
                fp = f_tvm(x_p) if model=="tvm" else f_cs(x_p)
                fn = f_tvm(x_n) if model=="tvm" else f_cs(x_n)
                
                if (not np.isfinite(fp)) or (not np.isfinite(fn)) or (fp <= 0) or (fn <= 0):
                    xx.append(np.nan); continue

                nuclear_int = fp*ep + fn*en - an*(n_p**2 + n_n**2) - 2.0*apn*n_p*n_n


                xx.append(eu + ed + nuclear_int)

            epsilon = np.asarray(xx, float)
            mask = np.isfinite(epsilon)
            eps_phys = epsilon[mask]

            if eps_phys.size == 0:
                energy_density_list.append(np.nan)
            else:
                energy_density_list.append(np.min(eps_phys))

        energy_density_arr = np.asarray(energy_density_list, float)

        plt.plot(
            rhob_list/rho0,
            energy_density_arr / rhob_list - mn,
            color=colors[iy],
            label=f"y={y:.1f}",lw=0.9
        )
        
    plt.ylim(-0.05, 0.5)
    plt.xlim(0, 5)
    plt.xlabel(r"$\rho_B [\rho_0]$")
    plt.ylabel(r"$\epsilon/\rho_B - m_n$")

    if isos_repulsion == 'eq':
        plt.title(r'$b_{pn} = b_n$')  
    elif   isos_repulsion == 'neq':
        plt.title(r'$b_{pn} \neq b_n$')  

    if model == 'tvm':
        if isos_repulsion == 'eq':
            plt.title(r' Trivirial model $b_{pn} = b_n$') 
        elif   isos_repulsion == 'neq':
            plt.title(r' Trivirial model $b_{pn} \neq b_n$') 

    elif model== 'cs':
        if isos_repulsion == 'eq':
            plt.title(r' Carnaghan Stirl model $b_{pn} = b_n$') 
        elif   isos_repulsion == 'neq':
            plt.title(r' Carnaghan Stirl  model $b_{pn} \neq b_n$') 

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
    })
    plt.legend()

    plt.tight_layout()
    if model == 'tvm':
        if isos_repulsion == 'eq':
            plt.savefig("./TVM_eq.png", dpi=90, bbox_inches="tight")
            plt.savefig("./TVM_eq.pdf", bbox_inches="tight") 
        elif   isos_repulsion == 'neq':
            plt.savefig("./TVM_neq.png", dpi=90, bbox_inches="tight")
            plt.savefig("./TVM_neq.pdf", bbox_inches="tight") 
    elif model== 'cs':
        if isos_repulsion == 'eq':
                plt.savefig("./cs_eq.png", dpi=90, bbox_inches="tight")
                plt.savefig("./cs_eq.pdf", bbox_inches="tight") 
        elif isos_repulsion == 'neq':
            plt.savefig("./cs_neq.png", dpi=90, bbox_inches="tight")
            plt.savefig("./cs_neq.pdf", bbox_inches="tight") 
    plt.show()
    plt.close()

main('eq','tvm')
main('neq','tvm')
main('eq','cs')
main('neq','cs')