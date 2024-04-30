lint: clean
	- pip install ruff codespell -q
	- ruff check --fix ullm/
	- codespell

format: lint
	- pip install ruff -q
	- ruff format ullm/

clean:
	- find . -iname "*__pycache__" | xargs rm -rf
	- find . -iname "*.pyc" | xargs rm -rf
	- rm cobertura.xml -f
	- rm testresult.xml -f
	- rm .coverage -f
	- rm .pytest_cache -rf
	- rm build/ -rf
	- rm dist -rf
	- rm *.egg-info -rf

test: clean
	- pip install -e .[test]
	- PYTHONPATH=. pytest -vvv --cov=ullm --cov-report term-missing --cov-fail-under=50 --cov-report xml:cobertura.xml --junitxml=testresult.xml tests/

lock-requirements:
	- pip install pip-tools -q
	- pip-compile --resolver=backtracking -U -o requirements.txt

deps: lock-requirements
	- pip-sync

build: lint test
	- python -m build
