# Contributing Guide

This guide covers the standards, guidelines, and processes for contributing to the Setu project. For detailed day-to-day development workflows, see the [Development Workflow Guide](development_workflow.md).

## Quick Start for New Contributors

### Prerequisites

- **CUDA 12.6+**: Setu has been tested with CUDA 12.6 on A100 and H100 GPUs
- **Python 3.12**: Required for development environment
- **Git**: For version control and submodule management

### Get Started in 5 Minutes

```shell
# Clone and setup
git clone -j8 https://github.com/project-vajra/setu
cd setu
mamba env create -f environment-dev.yml -p ./env
mamba activate ./env
make build

# Verify setup
python examples/offline_inference.py
```

For detailed setup instructions, see the [Development Workflow Guide](development_workflow.md#development-environment-setup).

## Contribution Process

### Before You Start

1. **Check Existing Issues**: Search for related issues or discussions
2. **Read the Documentation**: Familiarize yourself with the [design docs](../design/index.md)
3. **Understand the Codebase**: Start with smaller changes to learn the patterns

### Pull Request Process

1. **Fork and Branch**: Create a feature branch from main (naming convestion: `users/<user_name>/<feature_name>`)
2. **Follow Standards**: Adhere to coding standards (detailed below)
3. **Test Thoroughly**: Add tests and ensure all tests pass
4. **Document Changes**: Update documentation for user-facing changes and design patterns
5. **Submit PR**: Use clear description and link related issues, and name the PR appropriately

### Code Review Guidelines

**For Contributors**:
- Respond to feedback promptly and constructively
- Keep PRs focused and reasonably sized
- Be prepared to iterate on your changes

**For Reviewers**:
- Focus on correctness, performance, maintainability, and documentation
- Provide specific, actionable feedback
- Approve when ready, request changes when needed
- Aim to review within 2 business days

## Coding Standards

Setu maintains high code quality through automated formatting, comprehensive testing, and consistent style guides. 

**Python**: Follow PEP 8 with Black formatting (88-char lines), comprehensive type hints, and Setu-specific conventions. See [Python Style Guide](style_guides/python_style_guide.md) for details.

**C++**: Follow Google C++ Style Guide with Setu modifications, using modern C++20/23 features and structured logging. See [C++ Style Guide](style_guides/cpp_style_guide.md) for details.

## Testing and Quality Standards

All code changes must include appropriate tests and pass quality checks. See the [Development Workflow Guide](development_workflow.md#testing-strategy) for testing procedures.

## Documentation Requirements

- **Code Documentation**: Follow language-specific documentation standards in the style guides
- **API Changes**: Update API documentation for user-facing changes
- **Design Changes**: Update [design documents](../design/index.md) for architectural changes

Documentation is automatically generated from docstrings and comments.

## Performance and Architectural Considerations

- **Performance-Critical Changes**: Benchmark before/after and assess impact on critical paths
- **Native Integration**: Understand the Python/C++ hybrid execution model

See the [design documents](../design/index.md) for architectural guidance.

## Getting Help

### Resources

- Check the [Troubleshooting Guide](../troubleshooting.md) for common problems and solutions
- Report bugs on the [GitHub Issues page](https://github.com/project-vajra/setu/issues)
- Join the community discussion on [Discord](https://discord.gg/wjaSvGgsNN)
<!-- - Refer to the [API Reference](../usage/api_reference/index.md) for detailed documentation -->

### Best Practices for Getting Help

- Include relevant error messages and stack traces
- Provide minimal reproduction cases
- Share environment information when relevant
- Be specific about what you've already tried
