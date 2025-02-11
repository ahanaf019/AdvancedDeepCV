.PHONY: run install clean

run:
	rm -rf logs/supervised.*
	python -m supervised.main


clean:
	find . -type d -name "__pycache__" -exec rm -r {} +