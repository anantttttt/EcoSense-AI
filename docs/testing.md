# EcoSense AI Testing

## Test Case 1: Low-impact Profile

### Input

- Electricity: 100 kWh/month
- Transport: Walking / Cycling
- Waste: 2 kg/week
- Water: 100 litres/day

### Expected Behavior

The system should classify the profile toward the more sustainable
categories and provide positive sustainability guidance.

---

## Test Case 2: Moderate Profile

### Input

- Electricity: 250 kWh/month
- Transport: Motorcycle / Scooter
- Waste: 7 kg/week
- Water: 180 litres/day

### Expected Behavior

The system should identify a moderate sustainability profile and
highlight opportunities for improvement.

---

## Test Case 3: High-impact Profile

### Input

- Electricity: 400 kWh/month
- Transport: Petrol / Diesel Car
- Waste: 15 kg/week
- Water: 250 litres/day

### Expected Behavior

The system should identify a "Needs Improvement" profile and
highlight electricity, transportation, waste, and water as potential
impact areas.

---

## Testing Scope

Testing verifies:

- User input handling
- ML model prediction
- Sustainability classification
- Decision factors
- Impact analysis
- Recommendations
- Dashboard visualization
- Responsible AI information
- Application execution without runtime errors

## Dataset Limitation

The current model uses synthetic training data generated for
prototype demonstration. Testing therefore demonstrates system
functionality rather than validated real-world environmental accuracy.