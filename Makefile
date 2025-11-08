PY=python

.PHONY: smoke_test
smoke_test:
	$(PY) -m tests.smoke_test
