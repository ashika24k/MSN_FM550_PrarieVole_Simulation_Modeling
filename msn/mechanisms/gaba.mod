TITLE GABA synapse

COMMENT
Based on Examples 10.3 to 10.5 in The NEURON Book with parameters 
and `modulation()` function taken from Lindross's `gaba.mod` (ModelDB
#266775). That function allows to simulate dopamine and acetylcholine
modulation of this mechanism's conductance.

(2021) Antonio Gonzalez
ENDCOMMENT

NEURON {
    POINT_PROCESS gaba
    RANGE e, g, i
    RANGE damod, maxMod, level, max2, lev2
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
}

PARAMETER {
    e = -60 (mV)
    q = 2
    tau1 = 0.5 (ms) : Rise time constant.
    tau2 = 7.5 (ms) : Decay time constant. Must be greater than tau1.
    
    damod = 0
    maxMod = 1
    max2 = 1
    level = 0
    lev2 = 0
}

ASSIGNED {
    v (mV) 
    i (nA)
    g (uS)
    factor
}

STATE {
    a (uS)
    b (uS)
}

INITIAL {
    LOCAL tp
	a = 0
	b = 0
	tp = (tau1 * tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor = -exp(-tp/tau1) + exp(-tp/tau2)
    factor = 1/factor
}

BREAKPOINT {
    SOLVE states METHOD cnexp
	g = (b - a) * modulation()
	i = g * (v - e)
}

DERIVATIVE states {
	a' = -a/tau1 * q
	b' = -b/tau2 * q
}

NET_RECEIVE(weight (uS)) {
	a = a + weight * factor
	b = b + weight * factor
}

FUNCTION modulation() (1) {
    modulation = 1 + damod * (
        (maxMod - 1) * level +
        (max2 - 1) * lev2)
    if (modulation < 0) {
        modulation = 0
    }    
}