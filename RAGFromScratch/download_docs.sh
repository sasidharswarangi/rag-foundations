#!/usr/bin/env bash
# Downloads 10 diverse PDFs into ./docs for the RAG project.
# Run from the rag-project root: bash download_docs.sh

set -e
mkdir -p docs
cd docs

curl -sSL -o attention-is-all-you-need.pdf \
  https://arxiv.org/pdf/1706.03762

curl -sSL -o retrieval-augmented-generation.pdf \
  https://arxiv.org/pdf/2005.11401

curl -sSL -o bert.pdf \
  https://arxiv.org/pdf/1810.04805

curl -sSL -o resnet.pdf \
  https://arxiv.org/pdf/1512.03385

curl -sSL -o chain-of-thought-prompting.pdf \
  https://arxiv.org/pdf/2201.11903

curl -sSL -o constitutional-ai.pdf \
  https://arxiv.org/pdf/2212.08073

curl -sSL -o adam-optimizer.pdf \
  https://arxiv.org/pdf/1412.6980

curl -sSL -o us-constitution.pdf \
  https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/pdf/CDOC-110hdoc50.pdf

curl -sSL -o pride-and-prejudice.pdf \
  https://www.gutenberg.org/files/1342/1342-pdf.pdf

curl -sSL -o huckleberry-finn.pdf \
  https://www.gutenberg.org/files/76/76-pdf.pdf

cd ..
echo "Done. Files in ./docs:"
ls -la docs/
