SHELL := /bin/bash
PYTHON_ENV = . venv/bin/activate &&

build:
	python3 -m venv venv
	# $(PYTHON_ENV) pip3 install -r requirements.txt
	
start:
	$(PYTHON_ENV) python main.py

build_and_start: build start
