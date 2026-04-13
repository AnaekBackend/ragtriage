# Contributing to ragtriage

Thanks for your interest in contributing! This project was born from real frustration maintaining documentation at AttendanceBot, and we're excited to help other teams solve the same problem.

## How to Contribute

### Reporting Issues

- Use GitHub Issues
- Include sample queries (anonymized)
- Describe what you expected vs what happened
- Share your category taxonomy if you customized it

### Suggesting Enhancements

- Open an issue first to discuss
- Focus on actionable output (not just metrics)
- Consider how CS teams would use it

### Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit PR

## Development Setup

```bash
git clone https://github.com/yourusername/ragtriage.git
cd ragtriage
pip install -r requirements.txt
pip install -e .
```

## Areas We Need Help

### 1. More Lane Examples

The lane classifier (incident vs understanding vs spam) needs more examples for accuracy. If you have support queries that were misclassified, please share (anonymized).

### 2. Category Taxonomies

Our default categories (BILLING, LEAVE, TIMESHEET, etc.) are SaaS-focused. If you work in a different domain, suggest a taxonomy that works for your industry.

### 3. Prompt Improvements

The LLM prompts in `eval.py` and `analyze.py` can always be better. If you find edge cases where classification fails, help us improve the prompts.

### 4. Output Formats

Currently we output CSV and Markdown. Would your team benefit from:
- Notion integration?
- Linear/Jira ticket creation?
- Slack notifications?
- Something else?

### 5. Case Studies

If you use ragtriage at your company, we'd love a case study (can be anonymized). Real examples help others understand the value.

## Code Style

- Black for formatting
- Type hints encouraged
- Docstrings for public functions

## Testing

```bash
pytest tests/
```

## Questions?

- Open a GitHub Discussion for general questions
- Open an Issue for bugs
- Email: your.email@example.com

## Code of Conduct

Be kind, be constructive, focus on helping CS teams.
