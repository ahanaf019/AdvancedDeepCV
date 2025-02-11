.PHONY: run install clean

run:
	python -m supervised-learning.main


clean:
	find . -type d -name "__pycache__" -exec rm -r {} +