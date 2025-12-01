# Carbon Footprint Calculation

## Overview

Carbon footprint is calculated from two sources:
1. **Material Carbon Footprint**: Emissions from raw material production
2. **Transport Carbon Footprint**: Emissions from transporting products

## Material Carbon Footprint

### Formula
```
carbon_material = Σ (W × P_i × CF_i)
```

Where:
- `W` = Product weight (kg)
- `P_i` = Percentage of material i in product (0-1)
- `CF_i` = Carbon footprint factor for material i (kgCO2e/kg)

### Example
For a 0.5kg product with 80% cotton (2.5 kgCO2e/kg) and 20% polyester (5.5 kgCO2e/kg):
```
carbon_material = 0.5 × (0.8 × 2.5 + 0.2 × 5.5)
                = 0.5 × (2.0 + 1.1)
                = 1.55 kgCO2e
```

## Transport Carbon Footprint

### Multinomial Logit Model

Transport mode selection is modeled using a distance-dependent multinomial logit model.

#### Transport Modes
1. Road (reference mode)
2. Rail
3. Inland Waterway
4. Sea
5. Air

#### Utility Function
For each mode m:
```
U_m(D) = β0_m + β1_m × ln(D)
```

Where:
- `D` = Distance (km)
- `β0_m` = Mode-specific intercept
- `β1_m` = Log-distance coefficient

Road is the reference mode with U_road = 0.

#### Modal Split Probabilities
```
P_m(D) = exp(U_m(D)) / Σ exp(U_k(D))
```

### Emission Calculation

#### Formula
```
E(D) = (W / 1000) × D × (EF_weighted / 1000)
```

Where:
- `W` = Shipment weight (kg)
- `D` = Distance (km)
- `EF_weighted` = Weighted emission factor (gCO2e/tkm)

#### Weighted Emission Factor
```
EF_weighted = Σ P_m(D) × EF_m
```

### Example
For a 0.5kg product transported 15,000km:

Assuming modal split: Road 10%, Sea 80%, Air 10%
Emission factors: Road 62 gCO2e/tkm, Sea 8 gCO2e/tkm, Air 602 gCO2e/tkm

```
EF_weighted = 0.1 × 62 + 0.8 × 8 + 0.1 × 602 = 72.8 gCO2e/tkm
E = (0.5 / 1000) × 15000 × (72.8 / 1000) = 0.546 kgCO2e
```

## Data Sources

### Emission Factors
From `transport_emission_factors_generalised.csv`:
- Road: ~62 gCO2e/tkm
- Rail: ~22 gCO2e/tkm
- Inland Waterway: ~31 gCO2e/tkm
- Sea: ~8 gCO2e/tkm
- Air: ~602 gCO2e/tkm

### Utility Parameters
From `utility_attractiveness.csv`:
- Beta0 and Beta1 coefficients for each mode
- Derived from freight transport statistics
