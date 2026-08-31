TITLE Potassium current, calcium-dependent (BK)

COMMENT
Based on Section 9.6 in The NEURON Book with parameters taken from the
Lindroos's original bk.mod mechanism.

This mechanism is not modulated by DA or ACh so lacks that function.

(2021) Antonio Gonzalez
ENDCOMMENT

NEURON {
    SUFFIX bk
    USEION k READ ek WRITE ik
    USEION ca READ cai
    RANGE gbar, g, i
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S)  = (siemens)
    (mM) = (milli/liter)
    FARADAY = (faraday) (kilocoulombs)
    R = (k-mole) (joule/degC)
}

PARAMETER {
    gbar = 0 (S/cm2)
    d1 = 0.84
    d2 = 1
    k1 = 0.18 (mM)
    k2 = 0.011 (mM)
    abar = 0.48 (/ms)
    bbar = 0.28 (/ms)
}

ASSIGNED {
    v (mV)
    ek (mV) 
    ik (mA/cm2)
    cai (mM)
    i (mA/cm2)
    g (S/cm2)
    celsius (degC)
    tauo (ms)
    oinf
}

STATE {
    o (1) : Fraction of channels that are open
}

INITIAL {
    rate(v, cai)
    o = oinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * o
    i = g * (v - ek)
    ik = i
}

DERIVATIVE states {
    rate(v, cai)
    o' = (oinf - o) / tauo
}

FUNCTION alpha(Vm (mV), ca (mM)) (/ms) {
    alpha = abar * ca/(ca + exp1(k1, d1, Vm))
}

FUNCTION beta(Vm (mV), ca (mM)) (/ms) {
    beta = bbar / (1 + ca/exp1(k2, d2, Vm))
}

FUNCTION exp1(k (mM), d, Vm (mV)) (mM) {
    exp1 = k * exp(-2e-3 * d * FARADAY * Vm / R / (273.15 + celsius))
}

PROCEDURE rate(Vm (mV), ca (mM)) {
    LOCAL a, b
    a = alpha(Vm, ca)
    b = beta(Vm, ca)
    tauo = 1/(a + b)
    oinf = a/(a + b)
}