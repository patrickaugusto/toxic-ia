{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPH3S+h9Nz1+CHyKER/5wkH",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/patrickaugusto/wfghjkl/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 50,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OIGQ4AyHu5CK",
        "outputId": "521edafd-8b55-4a84-a204-e55a61a6da4b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                                       toxicity  severe_toxicity   obscene  identity_attack    insult    threat  sexual_explicit\n",
            "Você é um fracasso completo e só atrapalha os outros.  0.973017          0.00013  0.002648         0.000701  0.942556  0.000335         0.000332\n"
          ]
        }
      ],
      "source": [
        "from detoxify import Detoxify\n",
        "import pandas as pd\n",
        "\n",
        "# Texto para análise\n",
        "getJson = 'Você é um fracasso completo e só atrapalha os outros.'\n",
        "\n",
        "# Textos de entrada\n",
        "texto_de_entrada = [getJson]\n",
        "\n",
        "# Predições para diferentes modelos\n",
        "results = Detoxify('multilingual').predict(texto_de_entrada)\n",
        "\n",
        "# Exibir resultados\n",
        "df = pd.DataFrame(results, index=texto_de_entrada)\n",
        "\n",
        "print(df.to_string())\n",
        "\n"
      ]
    }
  ]
}