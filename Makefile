.PHONY: run install clean

run:
	python -m supervised.main


clean:
	find . -type d -name "__pycache__" -exec rm -r {} +