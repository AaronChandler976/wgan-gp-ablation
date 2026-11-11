# Google 10-min series variants (job SUBMIT events, 29 days)

Naive forecast (repeat last value) MAPE. The paper reports 10.58-11.49% model MAPE on Google-10m,
which would need a series with naive MAPE around 12-15%. No variant of the public trace is that smooth.

| variant | mean jobs / 10 min | naive MAPE, test part | naive MAPE, whole series |
| --- | --- | --- | --- |
| all SUBMIT events (used) | 160.0 | 27.62% | 25.44% |
| scheduling class 0 | 61.5 | 22.16% | 21.87% |
| scheduling class 1 | 51.2 | 47.24% | 43.23% |
| scheduling class 2 | 46.3 | 44.56% | 42.41% |
| scheduling class 3 | 1.0 | 71.34% | 82.80% |
| unique jobs (first SUBMIT) | 160.0 | 27.62% | 25.44% |
