# Makefile for compiling matplotlib plots into PDFs. Nothing fancy.

.PHONY: all clean

all:
	for f in *.py; do env python3 "$$f"; done

clean:
	$(RM) -- *.pdf
