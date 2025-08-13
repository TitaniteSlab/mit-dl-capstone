activate:
	source ./.venv/bin/activate

install:
	pip install -r requirements.txt

build-pex:
	pex -D . -r requirements.txt -m engine:main -o engine.pex

build-pyinstaller:
	pyinstaller --onefile --name engine1 --collect-all torch --collect-all chess --collect-all numpy src/engine.py