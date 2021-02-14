final_exp:
	python experiment_main_real.py
	python experiment_main_real_O.py
	python experiment_main_sl_1d.py
	python experiment_main_sl_1d_O.py
	python experiment_main_moa_1d.py
	python experiment_main_moa_1d_O.py
	python experiment_main_sl_1d_dynamic.py
	python experiment_main_sl_1d_dynamic_O.py

real:
	python experiment_main_real.py
	python experiment_main_real_O.py

ocwe:
	python experiment_main_sl_1d_O.py
	python experiment_main_moa_1d_O.py
	python experiment_main_sl_1d_dynamic_O.py
