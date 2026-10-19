"""Accessibility check for categorical palettes (CLAUDE_files/figure_style.md).

1. Contrast of each color against white (WCAG 2.1 ratio). Target: at least 3 for lines and markers.
2. Smallest CIEDE2000 difference between any two colors, with normal vision and simulated protan,
   deutan and tritan vision (Machado et al. 2009, severity 1). Target: at least 8 in every row.
3. Grayscale lightness L* of each color (information): colors with close L* are distinct in
   grayscale print only by line style or marker.
Usage: python check_palette.py "#bb7a03" "#5e3c99" ...
"""
import numpy as np, itertools, sys
# Machado et al. (2009) severity-1.0 matrices (linear RGB)
M={"normal":np.eye(3),
"protan":np.array([[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]]),
"deutan":np.array([[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]]),
"tritan":np.array([[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],[0.004733,0.691367,0.303900]])}
def lin(h):
    c=np.array([int(h[i:i+2],16) for i in (1,3,5)])/255
    return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def lab(l):
    l=np.clip(l,0,1)
    X=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])@l
    X=X/np.array([0.95047,1,1.08883])
    f=np.where(X>0.008856,np.cbrt(X),7.787*X+16/116)
    return np.array([116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2])])
def de2000(a,b):
    L1,a1,b1=a;L2,a2,b2=b
    C1=np.hypot(a1,b1);C2=np.hypot(a2,b2);Cb=(C1+C2)/2
    G=0.5*(1-np.sqrt(Cb**7/(Cb**7+25**7)));a1p=(1+G)*a1;a2p=(1+G)*a2
    C1p=np.hypot(a1p,b1);C2p=np.hypot(a2p,b2)
    h1=np.degrees(np.arctan2(b1,a1p))%360;h2=np.degrees(np.arctan2(b2,a2p))%360
    dL=L2-L1;dC=C2p-C1p;dh=h2-h1
    if C1p*C2p==0: dh=0
    elif dh>180: dh-=360
    elif dh<-180: dh+=360
    dH=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dh/2))
    Lb=(L1+L2)/2;Cbp=(C1p+C2p)/2
    hb=(h1+h2)/2 if abs(h1-h2)<=180 else (h1+h2+360)/2
    if C1p*C2p==0: hb=h1+h2
    T=1-0.17*np.cos(np.radians(hb-30))+0.24*np.cos(np.radians(2*hb))+0.32*np.cos(np.radians(3*hb+6))-0.20*np.cos(np.radians(4*hb-63))
    SL=1+0.015*(Lb-50)**2/np.sqrt(20+(Lb-50)**2);SC=1+0.045*Cbp;SH=1+0.015*Cbp*T
    RT=-2*np.sqrt(Cbp**7/(Cbp**7+25**7))*np.sin(np.radians(60*np.exp(-((hb-275)/25)**2)))
    return np.sqrt((dL/SL)**2+(dC/SC)**2+(dH/SH)**2+RT*(dC/SC)*(dH/SH))
def check(pal,show=True):
    worst=99
    for k,m in M.items():
        labs={h:lab(m@lin(h)) for h in pal}
        mn=min((de2000(labs[a],labs[b]),a,b) for a,b in itertools.combinations(pal,2))
        worst=min(worst,mn[0])
        if show: print(f"{k:7s} min ΔE {mn[0]:5.1f} ({mn[1]} vs {mn[2]})")
    return worst
def contrast(h):
    y=float(np.array([0.2126,0.7152,0.0722])@lin(h))
    return 1.05/(y+0.05)
def report(pal):
    for h in pal:
        c=contrast(h)
        print(f"{h}  contrast {c:5.2f} {'ok' if c>=3 else 'LOW'}  L* {lab(lin(h))[0]:5.1f}")
    return min(contrast(h) for h in pal), check(pal)
if __name__=="__main__":
    c,d=report(sys.argv[1:])
    print("PASS" if c>=3 and d>=8 else "FAIL")
