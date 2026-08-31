TITLE Potassium current, M-type (I_K(M))

COMMENT
A slowly activating, non-inactivating potassium current, muscarine-
sensitive. Modelled after Adams et al (1982).

A similar implentation of this current is that by Doron et al (2017),
available on ModelDB (#231427). Note, however, that their equations used
to calculate `alpha` and `beta` differ slightly from Equations 5a and 5b
in Adams et al.

The value of the temperature coefficient Q10 was taken from Doron's
model. The `modulation()` function is from Lindroos and Hellgren
Kotaleski (2020) (ModelDB #266775), and is used to simulate dopamine and
acetylcholine modulation of this mechanism.

References
----------
Adams PR, Brown DA & Constanti A (1982). M-currents and other potassium
currents in bullfrog sympathetic neurones. J Physiol 330: 537-572.

Lindroos R & Hellgren Kotaleski J (2020). Predicting complex spikes in
striatal projection neurons of the direct pathway following
neuromodulation by acetylcholine and dopamine. Eur J Neurosci
(https://doi.org/10.1111/ejn.14891).


(2021) Antonio Gonzalez
ENDCOMMENT

NEURON {
    SUFFIX km
    USEION k READ ek WRITE ik
    RANGE gbar, g, i
    RANGE damod, maxMod, level, max2, lev2
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S)  = (siemens)
    : F = (faraday) (kilocoulombs)
    : R = (k-mole) (joule/degC)
}

PARAMETER {
    gbar = 0.00001 (S/cm2)
    damod = 0
    maxMod = 1
    max2 = 1
    level = 0
    lev2 = 0
    celsius (degC)
    q10 = 2.3
    q10temp = 22 (degC) : Original temperature, from Adams et al (1982).
}

ASSIGNED {
    v (mV)
    ek (mV) 
    ik (mA/cm2)
    i (mA/cm2)
    g (S/cm2)
    tadj (1)
    taum (ms)
    minf (1)
}

STATE {
    m (1)
}

INITIAL {
    tadj = q10^((celsius - q10temp)/(10(degC))) : Temperature-adjusting factor
    rates(v)
    m = minf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * m * modulation()
    i = g * (v - ek)
    ik = i
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / taum
}

PROCEDURE rates (Vm (mV)) {
    : After Equations 4, 5a and 5b in Adams et al (1982). I simplified
    : those equations by reducing the constants used there to the
    : parameters `ktau1` and `ktau2`. Thus,
    :
    :   ktau1 = 1/((z * F)/(2 * R * T))
    :   ktau2 = 1/((-z * F)/(2 * R * T))
    : 
    : where
    :   z = 2.5 (1)
    :   F = 96.480 (kilocoul/mole)
    :   R = 8.314 (joule/degC)
    :   T = 273.15 + 22 (celsius)
    LOCAL vtau, ktau1, ktau2, alpha, beta
    vtau = -35 (mV)
    ktau1 = 20.3 (mV)
    ktau2 = -20.3 (mV)
    alpha = 3.3(/s) * exp((Vm - vtau)/ktau1)
    beta = 3.3(/s) * exp((Vm - vtau)/ktau2)
    taum = 1/(alpha + beta) * (1000)
    taum = taum/tadj
    minf = alpha/(alpha + beta)
}

FUNCTION modulation() {
    modulation = 1 + damod * (
        (maxMod - 1) * level +
        (max2 - 1) * lev2)
    if (modulation < 0) {
        modulation = 0
    }    
}