# Compiles the presentation
# In case of errors, check the log file: cache/presentation.log

CACHEDIR = cache
GVDIR = graphviz
PYPLOTDIR = pyplot
LATEX = xelatex
JOBNAME = presentation
LXFLAGS = -interaction batchmode -8bit -shell-escape -output-directory "$(CACHEDIR)" -jobname $(JOBNAME)

.PHONY: directories all graphviz pyplot main clean

all: directories graphviz pyplot main

directories:
	mkdir -p -- $(CACHEDIR)

graphviz:
	$(MAKE) -C $(GVDIR)

pyplot:
	$(MAKE) -C $(PYPLOTDIR)

main: main.tex
	@echo Starting build...
	$(LATEX) $(LXFLAGS) -- $^
	$(LATEX) $(LXFLAGS) -- $^
	$(LATEX) $(LXFLAGS) -- $^

	mv -- $(CACHEDIR)/$(JOBNAME).pdf .
	@echo Build successful.

clean:
	$(MAKE) -C $(GVDIR)
	$(RM) -r -- $(JOBNAME).pdf $(CACHEDIR)/
