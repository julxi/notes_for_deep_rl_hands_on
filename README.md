# Notes for Deep Reinforcement Learning Hands-On

This repository contains my study notes and updated code implementations for **"Deep Reinforcement Learning Hands-On"** by Maxim Lapan. The materials include both executable code examples and quarto code for the notes, which are published as [gitHub page](https://julxi.github.io/notes_for_deep_rl_hands_on/).

## 📌 Overview
- **purpose**: supplemental materials for Maxim Lapan's DRL book with working code and expanded explanations
- **features**: 
  - updated chapter code implementations (with fixes for library changes, e.g., gymnasium[atari])
  - Quarto notebooks for own experimentation
- **target audience**: learners working through the book who might benefit from additional resources

## 📚 Project Structure
```
├── code/ # chapter-specific code examples
│ └── chapter_X/ # each chapter has its own folder
├── quarto/ # quarto chapters
├── requirements.txt # python dependencies for code examples
└── requirements_quarto.txt # dependencies for rendering  notes
```

## 🛠️ Installation

### For Code Examples
```bash
# clone the repository
git clone https://github.com/julxi/notes_for_deep_rl_hands_on.git

# install core dependencies
pip install -r requirements.txt

# run chapter-specific code, e.g.,
python code/chapter_04/01_cartpole.py
```


### For rendering Quarto notes

```bash
# 1. install Quarto CLI: https://quarto.org/docs/install/
# 2. install dependencies (in repository root)
pip install -r requirements.txt
pip install -r requirements_quarto.txt

# 3. render documentation
quarto render
```

    💡 Tip: after rendering, you can interact with the notes and execute embedded Python code cells for hands-on learning

## 📖 GitHub Page

View the rendered notes and examples at:
https://julxi.github.io/notes_for_deep_rl_hands_on/


## 🤝 Contributing

feedback and improvements are welcome!  
contact: Julian Bitterlich (bitt.j@protonmail.com)  
suggested contributions:

- zug fixes for code examples
- additional chapter explanations/clarifications
- improved documentation

## 📄 License

MIT License - see [LICENSE](LICENSE) file