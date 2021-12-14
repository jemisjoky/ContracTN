.PHONY: clean
clean:
	rm -rf contractn.egg-info
	rm -rf .pytest_cache/

# .PHONY: dev-requirements
# dev-requirements: dev_requirements.txt
# 	pip install -r dev_requirements.txt

# .PHONY: dist
# dist:
# 	python setup.py

# .PHONY: docs
# docs:
# 	make -C docs html

.PHONY: format
format:
	black contractn/

.PHONY: install
install:
	pip install -e .

.PHONY: requirements
requirements: requirements.txt
	pip install -r requirements.txt

.PHONY: style
style:
	flake8 contractn/

.PHONY: test
test:
	pytest -x contractn/tests
