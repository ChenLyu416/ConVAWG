# VAWG Crime Types - Quick Reference

## Complete Crime Type Taxonomy (25 Categories)

### Physical & Direct Harm
| Code | Description |
|------|-------------|
| `physical` | Physical assault, violence, bodily harm |

### Sexual Offences
| Code | Description |
|------|-------------|
| `sexual` | General sexual offences (use specific types below when applicable) |
| `rape` | Non-consensual penis penetration (vagina, mouth, anus) |
| `sexual_assault` | Non-consensual sexual touching (no penetration) |
| `assault_by_penetration` | Non-consensual penetration with body/object (not penis) |
| `child_sexual_abuse` | Sexual offences against persons under 18 |

### Psychological & Emotional Harm
| Code | Description |
|------|-------------|
| `psychological` | Psychological abuse, intimidation, fear |
| `emotional` | Emotional abuse, humiliation, degradation |
| `coercive_control` | Pattern of controlling/coercive behaviour (Serious Crime Act 2015) |

### Economic Harm
| Code | Description |
|------|-------------|
| `economic` | Economic abuse, financial control |
| `financial` | Exploitation of financial resources, property |

### Stalking & Harassment
| Code | Description |
|------|-------------|
| `stalking` | Fixated, Obsessive, Unwanted, Repeated (FOUR) intrusive behaviour |
| `harassment` | Repeated behaviour causing alarm/distress (2+ occasions) |
| `cyberstalking` | Digital stalking via social media, email, tracking apps |
| `digital` | Other digital/online abuse |

### Child Abuse (Non-Sexual)
| Code | Description |
|------|-------------|
| `child_abuse_non_sexual` | Non-sexual maltreatment of persons under 18 |
| `neglect` | Failure to provide care or prevent harm |

### Honour-Based & Cultural Practices
| Code | Description |
|------|-------------|
| `honor_based` | HBA - violence to protect/defend family/community honour |
| `forced_marriage` | Forcing marriage without consent or capacity |
| `fgm` | Female genital mutilation (WHO Types 1-4) |

### Exploitation & Trafficking
| Code | Description |
|------|-------------|
| `trafficking` | Human trafficking, sexual exploitation |
| `exploitation` | Sexual or labour exploitation |
| `prostitution_related` | Controlling prostitution for gain, related offences |

### Publications
| Code | Description |
|------|-------------|
| `obscene_publications` | Distribution of obscene/extreme pornographic material |

### Other
| Code | Description |
|------|-------------|
| `other` | Other VAWG forms not listed above |

---

## Legal Framework Quick Reference

| Crime Type | Primary Legislation |
|------------|-------------------|
| `rape`, `sexual_assault`, `assault_by_penetration` | Sexual Offences Act 2003 |
| `coercive_control` | Serious Crime Act 2015, Section 76 |
| `obscene_publications` | Obscene Publications Act 1959 |
| `forced_marriage` | Anti-social Behaviour, Crime and Policing Act 2014 |
| `stalking`, `harassment` | Protection from Harassment Act 1997 |

---

## Common Combinations

Real cases often involve multiple crime types. Common combinations:

**Intimate Partner Violence:**
- `physical` + `psychological` + `coercive_control`
- `sexual_assault` + `coercive_control` + `financial`

**Digital Abuse:**
- `cyberstalking` + `harassment` + `obscene_publications`
- `coercive_control` + `digital` + `stalking`

**Child Abuse:**
- `child_abuse_non_sexual` + `neglect` + `psychological`
- `child_sexual_abuse` + `coercive_control` + `digital`

**Honour-Based Violence:**
- `honor_based` + `forced_marriage` + `physical`
- `fgm` + `honor_based` + `psychological`

**Exploitation:**
- `trafficking` + `prostitution_related` + `coercive_control`
- `exploitation` + `sexual` + `financial`

---

## Decision Tree for Sexual Offences

```
Sexual offence occurred?
├─ Victim under 18? → child_sexual_abuse
└─ Victim 18+
   ├─ Penis penetration? → rape
   ├─ Other penetration? → assault_by_penetration
   ├─ Sexual touching? → sexual_assault
   └─ General/multiple → sexual
```

---

## Decision Tree for Digital Abuse

```
Digital/online component?
├─ FOUR pattern stalking? → cyberstalking
├─ Obscene images shared? → obscene_publications
├─ Digital control/monitoring? → coercive_control + digital
└─ Other online abuse → digital
```

---

## Key Acronyms

- **VAWG** - Violence Against Women and Girls
- **CPS** - Crown Prosecution Service
- **HBA** - Honour-Based Abuse
- **FGM** - Female Genital Mutilation
- **FOUR** - Fixated, Obsessive, Unwanted, Repeated (stalking criteria)
- **CCB** - Controlling or Coercive Behaviour
- **DHR** - Domestic Homicide Review

---

## Usage in Code

```python
from schemas.vawg_scenario import ViolenceType

# Access all valid types
all_types = [v.value for v in ViolenceType]

# Check if value is valid
if "rape" in ViolenceType.__members__.values():
    print("Valid crime type")

# Filter scenarios by category
sexual_offences = [
    ViolenceType.rape,
    ViolenceType.sexual_assault,
    ViolenceType.assault_by_penetration,
    ViolenceType.child_sexual_abuse
]
```

---

**For complete definitions, see:** [`CPS_VAWG_DEFINITIONS.md`](./CPS_VAWG_DEFINITIONS.md)
