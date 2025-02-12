.PHONY: run install clean

run:
	rm -rf logs/supervised.*
	python -m supervised.main_classification


clean:
	find . -type d -name "__pycache__" -exec rm -r {} +