# CiCi's Paper Laboratory

## 1. Papers Included

- [x] [ID 001](./src/paper_lab/paper_001/README.md): Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. **Attention Is All You Need.**

- [ ] [ID 002](./src/paper_lab/paper_002/README.md): Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**

- [ ] [ID 003](./src/paper_lab/paper_003/README.md):Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. **Improving language understanding by generative pre-training.**

- [ ] [ID 004](./src/paper_lab/paper_004/README.md): Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. **Language Models are Unsupervised Multitask Learners.**

- [ ] [ID 005](./src/paper_lab/paper_005/README.md): Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2020. **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.**

- [ ] [ID 006](./src/paper_lab/paper_006/README.md): Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. **Language Models are Few-Shot Learners.**

## 2. Prerequisites

### Requirements

- Python 3.13+
- [astral-uv](https://github.com/astral-sh/uv)

### Installation/Setup

To set up the environment, follow these steps:

1. Clone and navigate to the repository:

   ```bash
   git clone https://github.com/weecici/paper-lab
   cd paper-lab
   ```

2. Let astral-uv handle the environment setup:

   ```bash
   uv sync
   ```

3. If you want to run test cases, you can execute:

   ```bash
   uv run pytest <path/to/test/cases>
   ```

## 3. License

Please refer to the [LICENSE](LICENSE) for details on the licensing of this project.
