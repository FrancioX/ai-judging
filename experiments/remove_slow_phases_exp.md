 These are the results before removing the slow phases which seem to not add much to tracking

 ========================================================================================================================
  COMBINED EVALUATION — Detection vs Tracking  (threshold=50.0px)
========================================================================================================================
  Video                                            Det%  DetErr Det<Th CorPer  IDs  LRun  TrkErr Trk<Th   HOTA  Imprv
  -------------------------------------------------------------------------------------------------------------------
  12_Ski Men_Gabin Leonard_26_France_74_          97.4%    30.0  95.5%  74.1%    7 0.298     2.8 100.0%  0.969   +91%
  16_Ski Men_Loris Gonzalez_10_Switzerland_71_    88.5%   144.0  61.3%  47.1%   18 0.121     3.9  98.6%  0.955   +97%
  15_Ski Men_Tibo Mantero_14_Switzerland_71.33_   78.2%    36.1  85.5%  89.3%   29 0.150     4.6  98.8%  0.950   +87%
  17_Ski Men_Coen Bennie-Faull_46_Australia_68.   89.9%   108.2  62.6%  90.6%   28 0.086     4.8  99.4%  0.949   +96%
  14_Ski Men_Lucas Daines_60_Canada_73.33_        82.0%    16.1  98.3% 100.0%   13 0.305     9.3  93.9%  0.909   +42%
  11_Snowboard Men_Nicolas Lagger_73_Switzerlan   88.0%     8.3  98.5%  98.3%   13 0.233    10.9  94.5%  0.902   -31%
  14_Snowboard Men_Taketo Kinoshita_71_Japan_48   96.6%     8.3  99.1%  92.6%    5 0.385    33.5  89.6%  0.876  -304%
  2_Ski Men_Andreas Bakke_24_Norway_89            93.8%    17.9  98.2%  99.5%   11 0.224    18.1  94.3%  0.869    -1%
  15_Snowboard Men_Theodor Salen_59_Norway_45_    75.7%    13.6  97.5%  89.7%   26 0.078    17.8  88.5%  0.856   -31%
  16_Snowboard Men_Jonatan Laland_61_Norway_40_   83.8%    17.6  95.6%  87.7%   22 0.093    60.6  81.2%  0.771  -244%
  1_Ski Men_Arno Vuarnier_58_Switzerland_91.33    63.3%    74.5  85.6%  96.1%   23 0.197    53.9  76.4%  0.704   +28%
  12_Snowboard Men_Adriano Cardillo_63_Switzerl   79.5%     8.5  99.4% 100.0%   15 0.185    46.1  71.3%  0.681  -442%
  17_Snowboard Men_Quentin Puydenus_53_France_3   61.0%    70.4  83.7%  96.5%   29 0.079    49.3  69.9%  0.668   +30%
  3_Ski Men_Lach Powell_8_New Zealand_86          87.1%    19.1  97.9%  89.0%   11 0.283    61.2  68.5%  0.648  -220%
  11_Ski Men_Emile Peizerat_76_France_74.83_      78.1%     9.5  99.4%  96.5%   18 0.153    83.4  61.0%  0.577  -778%
  13_Ski Men_Maximilien Michel_28_France_73.67_   83.3%    32.8  94.5%  88.6%   29 0.107   197.2  59.7%  0.571  -501%
  10_Snowboard Men_Cedric Giraudeau_67_France_5   80.2%   131.3  11.3%  77.3%   17 0.252   122.0  15.7%  0.166    +7%
  10_Ski Men_Jordan Koch_56_Switzerland_75_       71.5%   234.3   5.0%  45.5%   25 0.103   253.3   2.6%  0.034    -8%
  -------------------------------------------------------------------------------------------------------------------
  AVERAGE                                         82.1%    54.5         86.6%               57.4         0.725
========================================================================================================================
  Videos: 18  |  Det%=detection rate  |  DetErr/TrkErr=mean center error
  CorPer=correct person in multi-person frames  |  IDs=ByteTrack fragmentation
  LRun=longest run ratio  |  Imprv=tracking error reduction vs detection
========================================================================================================================

## After removing slow phases (OF disabled, identity guard off, no smoothing)

Date: 2026-03-11
Config changes: `optical_flow_method: "none"`, `identity_guard_enabled: false` (smooth_window and cmc already off)

========================================================================================================================
  COMBINED EVALUATION — Detection vs Tracking  (threshold=50.0px)
========================================================================================================================
  Video                                            Det%  DetErr Det<Th CorPer  IDs  LRun  TrkErr Trk<Th   HOTA  Imprv
  -------------------------------------------------------------------------------------------------------------------
  11_Snowboard Men_Nicolas Lagger_73_Switzerlan   88.0%     8.3  98.5%  98.3%   13 0.233    10.9  94.5%  0.902   -31%
  14_Snowboard Men_Taketo Kinoshita_71_Japan_48   96.6%     8.3  99.1%  92.6%    5 0.385    33.5  89.6%  0.876  -304%
  15_Ski Men_Tibo Mantero_14_Switzerland_71.33_   87.3%    13.7  92.5%  87.0%    9 0.319    22.1  91.2%  0.871   -61%
  11_Ski Men_Emile Peizerat_76_France_74.83_      78.1%     9.5  99.4%  96.5%   18 0.153    15.5  94.2%  0.865   -63%
  15_Snowboard Men_Theodor Salen_59_Norway_45_    75.7%    13.6  97.5%  89.7%   26 0.078    17.8  88.5%  0.856   -31%
  3_Ski Men_Lach Powell_8_New Zealand_86          87.1%    19.1  97.9%  89.0%   11 0.283    48.8  91.6%  0.849  -155%
  12_Ski Men_Gabin Leonard_26_France_74_          97.4%    30.0  95.5%  74.1%    7 0.298    78.7  86.9%  0.839  -162%
  14_Ski Men_Lucas Daines_60_Canada_73.33_        82.0%    16.1  98.3% 100.0%   13 0.305    80.7  83.3%  0.790  -401%
  16_Snowboard Men_Jonatan Laland_61_Norway_40_   83.8%    17.6  95.6%  87.7%   22 0.093    60.6  81.2%  0.771  -244%
  2_Ski Men_Andreas Bakke_24_Norway_89            93.8%    17.9  98.2%  99.5%   11 0.224    92.4  79.2%  0.728  -416%
  13_Ski Men_Maximilien Michel_28_France_73.67_   83.3%    32.8  94.5%  88.6%   29 0.107   120.1  77.5%  0.724  -266%
  12_Snowboard Men_Adriano Cardillo_63_Switzerl   79.5%     8.5  99.4% 100.0%   15 0.185    46.1  71.3%  0.681  -442%
  17_Snowboard Men_Quentin Puydenus_53_France_3   61.0%    70.4  83.7%  96.5%   29 0.079    49.3  69.9%  0.668   +30%
  17_Ski Men_Coen Bennie-Faull_46_Australia_68.   89.9%   108.2  62.6%  90.6%   28 0.086   167.8  57.9%  0.553   -55%
  16_Ski Men_Loris Gonzalez_10_Switzerland_71_    88.5%   144.0  61.3%  47.1%   18 0.121   188.2  57.6%  0.547   -31%
  1_Ski Men_Arno Vuarnier_58_Switzerland_91.33    63.3%    74.5  85.6%  96.1%   23 0.197   162.5  60.9%  0.542  -118%
  10_Snowboard Men_Cedric Giraudeau_67_France_5   80.2%   131.3  11.3%  77.3%   17 0.252   122.0  15.7%  0.166    +7%
  10_Ski Men_Jordan Koch_56_Switzerland_75_       71.5%   234.3   5.0%  45.5%   25 0.103   260.4   3.8%  0.036   -11%
  -------------------------------------------------------------------------------------------------------------------
  AVERAGE                                         82.6%    53.2         86.4%               87.6         0.681
========================================================================================================================

## Comparison: Before vs After removing slow phases

| Metric              | Before (OF+guard) | After (no OF) | Delta    |
|---------------------|--------------------|---------------|----------|
| Avg TrkErr (px)     | 57.4               | 87.6          | +30.2    |
| Avg HOTA            | 0.725              | 0.681         | -0.044   |

### Per-video HOTA delta (biggest regressions)

| Video                   | Before | After | Delta  |
|-------------------------|--------|-------|--------|
| 12_Gabin Leonard        | 0.969  | 0.839 | -0.130 |
| 16_Loris Gonzalez       | 0.955  | 0.547 | -0.408 |
| 15_Tibo Mantero         | 0.950  | 0.871 | -0.079 |
| 17_Coen Bennie-Faull    | 0.949  | 0.553 | -0.396 |
| 14_Lucas Daines         | 0.909  | 0.790 | -0.119 |
| 2_Andreas Bakke         | 0.869  | 0.728 | -0.141 |
| 11_Emile Peizerat       | 0.577  | 0.865 | +0.288 |
| 13_Maximilien Michel    | 0.571  | 0.724 | +0.153 |
| 3_Lach Powell           | 0.648  | 0.849 | +0.201 |

### Conclusion

The OF phases have a **mixed** effect across the full dataset. The ablation on Tibo Mantero
was misleading — it happened to be a video where OF contributed little. Across all 18 videos:

- **6 videos got significantly worse** without OF (Loris -0.408, Coen -0.396, Andreas -0.141, Gabin -0.130, Lucas -0.119, Tibo -0.079)
- **3 videos got significantly better** without OF (Emile +0.288, Lach +0.201, Maximilien +0.153)
- **9 videos were unchanged** (snowboard videos had no re-run, so same results)

The OF phases clearly help on some videos (particularly those with large camera motion or long
detection gaps) but hurt on others (likely due to identity guard over-rejecting valid detections).
Average HOTA dropped from 0.725 to 0.681, and average TrkErr increased from 57.4 to 87.6px.

**Verdict: Keep OF phases enabled.** Reverting config changes.