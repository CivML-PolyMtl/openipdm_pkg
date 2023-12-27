# %%
from ssm_kr import SSM_KR
import numpy as np
# create an instance of the model
det_model = SSM_KR()

# define inspectors errors (ID, bias, standard deviation)
inspector_std= np.array([[ 0. ,  0. ,  2. ],
                        [ 1. ,  0. ,  4. ],
                        [ 2. ,  3. ,  4. ],
                        [ 3. , -3. ,  4. ],
                        [ 4. ,  3. ,  1.5],
                        [ 5. , -3. ,  1.5]])

# define the actions (0: do nothing, 1: perventive maintenance, 2: routine maintenance, 3: repairs, 4: replace)
actions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

# define which inspeptors performed the inspection over time
inspectors = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

# define the total number of years
total_years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028])

# define the inspections performed over time
y = np.array([[  np.nan, 88.75,   np.nan, 86.25,   np.nan, 85.  ,   np.nan, 85.  ,   np.nan,
                    np.nan,   np.nan,   np.nan,   np.nan]])

# run the model
df_cond, df_speed = det_model.ssm_kr_predict(y=y, total_years=total_years, inspector_std=inspector_std, inspector=inspectors, Actions=actions)

# plot the results
plot_alt = det_model.plot_results(df_cond, df_speed)

plot_alt



