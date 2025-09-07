# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build
LOG_DIR      = logs

# Timestamp for log files
TIMESTAMP = $(shell date +%Y%m%d_%H%M%S)

# Colors and formatting (similar to main Makefile)
_IS_TTY := $(shell test -t 1 && echo true)
ifeq ($(_IS_TTY),true)
  ifndef NO_COLOR
    BOLD    := \033[1m
    GREEN   := \033[0;32m
    YELLOW  := \033[0;33m
    BLUE    := \033[0;94m
    CYAN    := \033[0;36m
    RESET   := \033[0m
    ECHO = /bin/echo -e
  endif
endif

ifndef ECHO
    ECHO = /bin/echo
endif

# Emojis
SETUP_ICON="⚙️ "
DOCS_ICON="📚"
SUCCESS_ICON="✅"
PARTY_ICON="🎉"
INFO_ICON="ℹ️ "

.PHONY: help Makefile clean/doxygen clean/html clean build/doxygen setup/dependencies build/html build/html-fast _setup_dirs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(ECHO) ""
	@$(ECHO) "$(BOLD)Additional Documentation Targets:$(RESET)"
	@$(ECHO) "  $(CYAN)setup/dependencies$(RESET)         Install documentation dependencies"
	@$(ECHO) "  $(CYAN)build/doxygen$(RESET)            Generate Doxygen XML output"
	@$(ECHO) "  $(CYAN)build/html$(RESET)          Build HTML documentation"
	@$(ECHO) "  $(CYAN)build/html-fast$(RESET)          Quick build without Doxygen"
	@$(ECHO) "  $(CYAN)serve$(RESET)           Live reload server"
	@$(ECHO) "  $(CYAN)clean/doxygen$(RESET)      Clean Doxygen output"
	@$(ECHO) "  $(CYAN)clean/html$(RESET)      Clean HTML output"
	@$(ECHO) "  $(CYAN)clean$(RESET)      Clean all output"

# Create necessary directories
_setup_dirs:
	@$(ECHO) "$(BLUE)$(SETUP_ICON) Ensuring directories exist: $(LOG_DIR)$(RESET)"
	@mkdir -p $(LOG_DIR)

# Install documentation build/dependencies
setup/dependencies: _setup_dirs ## install documentation dependencies
	@$(ECHO) "$(BOLD)$(SETUP_ICON) Installing Documentation Dependencies...$(RESET)"
	@$(ECHO) "$(BLUE)$(INFO_ICON) Using requirements.txt for documentation dependencies$(RESET)"
	@$(ECHO) "$(BLUE)$(INFO_ICON) Full log: $(LOG_DIR)/pip_install_docs_$(TIMESTAMP).log$(RESET)"
	@pip install -r requirements.txt 2>&1 | tee "$(LOG_DIR)/pip_install_docs_$(TIMESTAMP).log"
	@$(ECHO) "$(GREEN)$(SUCCESS_ICON) Documentation dependencies installed successfully $(PARTY_ICON)$(RESET)"

# Generate Doxygen XML output before build/building Sphinx docs
build/doxygen: _setup_dirs
	@$(ECHO) "$(BLUE)🔧 Generating Doxygen XML output...$(RESET)"
	@$(ECHO) "$(BLUE)$(INFO_ICON) Doxygen log: $(LOG_DIR)/doxygen_$(TIMESTAMP).log$(RESET)"
	@doxygen Doxyfile 2>&1 | tee "$(LOG_DIR)/doxygen_$(TIMESTAMP).log"
	@$(ECHO) "$(GREEN)✅ Doxygen XML generated in doxygen_output/xml/$(RESET)"

# Clean Doxygen output
clean/doxygen:
	@$(ECHO) "$(YELLOW)🧹 Cleaning Doxygen output...$(RESET)"
	@rm -rf doxygen_output/

# Clean HTML output
clean/html:
	@$(ECHO) "$(YELLOW)🧹 Cleaning HTML output...$(RESET)"
	@rm -rf _build/html/

# Clean everything including Doxygen
clean: clean/doxygen clean/html

# Build HTML with Doxygen dependency
build/html: setup/dependencies build/doxygen
	@$(ECHO) "$(BOLD)$(DOCS_ICON) Building HTML Documentation...$(RESET)"
	@$(ECHO) "$(BLUE)$(INFO_ICON) Sphinx log: $(LOG_DIR)/sphinx_html_$(TIMESTAMP).log$(RESET)"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee "$(LOG_DIR)/sphinx_html_$(TIMESTAMP).log"
	@$(ECHO) "$(GREEN)$(SUCCESS_ICON) HTML documentation built successfully$(RESET)"

# Live reload server
serve: setup/dependencies build/doxygen
	@$(ECHO) "$(BOLD)$(DOCS_ICON) Starting Live Reload Server...$(RESET)"
	@$(ECHO) "$(BLUE)$(INFO_ICON) Documentation will be available at http://127.0.0.1:8000$(RESET)"
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--watch ../setu \
		--ignore '**/*.pyc' \
		--ignore '**/__pycache__' \
		--ignore '**/.*' \
		--port 8000

# Quick build without Doxygen (for iteration)
build/html-fast: setup/dependencies _setup_dirs
	@$(ECHO) "$(BOLD)$(DOCS_ICON) Building HTML Documentation (Fast - No Doxygen)...$(RESET)"
	@$(ECHO) "$(BLUE)$(INFO_ICON) Fast build log: $(LOG_DIR)/sphinx_fast_$(TIMESTAMP).log$(RESET)"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee "$(LOG_DIR)/sphinx_fast_$(TIMESTAMP).log"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: setup/dependencies build/doxygen Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)