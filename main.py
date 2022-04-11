{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_ey8HjE0GVZA",
        "outputId": "21257270-e3d7-4121-fc2a-3654fb1b5637"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dI_O6cKGIQmj",
        "outputId": "2ccc4b67-477d-419a-8fcd-0810fe13f009"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "fatal: destination path 'Progetto-Tesi' already exists and is not an empty directory.\n"
          ]
        }
      ],
      "source": [
        "!git clone https://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@github.com/StefanoBavaro/Progetto-Tesi.git"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J6M4XlTQIdug",
        "outputId": "3f7babd3-d8c6-4878-ada1-35d0302c515f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/Progetto-Tesi\n"
          ]
        }
      ],
      "source": [
        "%cd Progetto-Tesi/"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I7MZE5rhJLRG",
        "outputId": "6fece85f-4e97-434e-9914-22b26d6eded8"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[0m\u001b[01;34m.\u001b[0m/  \u001b[01;34m..\u001b[0m/  \u001b[01;34m.config\u001b[0m/  \u001b[01;34mProgetto-Tesi\u001b[0m/  \u001b[01;34msample_data\u001b[0m/\n"
          ]
        }
      ],
      "source": [
        "%ls -a"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "o_IOy_A89jzu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sXnDmXR7RDr2",
        "outputId": "f60c0a07-487a-4f49-9acc-e56b554924c9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found GPU at: /device:GPU:0\n"
          ]
        }
      ],
      "source": [
        "%tensorflow_version 2.x\n",
        "import tensorflow as tf\n",
        "device_name = tf.test.gpu_device_name()\n",
        "if device_name != '/device:GPU:0':\n",
        "  raise SystemError('GPU device not found')\n",
        "print('Found GPU at: {}'.format(device_name))"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aycNp9Rzdm9B",
        "outputId": "693e4a56-72aa-416f-91cc-94d8d61ea258"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "*** Please tell me who you are.\n",
            "\n",
            "Run\n",
            "\n",
            "  git config --global user.email \"you@example.com\"\n",
            "  git config --global user.name \"Your Name\"\n",
            "\n",
            "to set your account's default identity.\n",
            "Omit --global to set the identity only in this repository.\n",
            "\n",
            "fatal: unable to auto-detect email address (got 'root@18783349822b.(none)')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xozknvD4J6PK",
        "outputId": "5dad2b85-4c25-48df-aad0-d1d5e95d484f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Processing CSV...\n",
            "NUMERO TRACCE = 782\n",
            "NUMERO EVENTI = 13120\n",
            "NUMERO EVENTI DOPO UPDATE= 13902\n",
            "17\n",
            "2\n",
            "Done: updated CSV created\n",
            "  0% 0/20 [00:00<?, ?it/s, best loss: ?]2022-04-11 18:23:20.176443: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.\n",
            "Model: \"model\"\n",
            "\n",
            "__________________________________________________________________________________________________\n",
            "\n",
            " Layer (type)                   Output Shape         Param #     Connected to                     \n",
            "\n",
            "==================================================================================================\n",
            "\n",
            " input_act (InputLayer)         [(None, 4)]          0           []                               \n",
            "\n",
            " embedding (Embedding)          (None, 4, 23)        414         ['input_act[0][0]']              \n",
            "\n",
            " lstm (LSTM)                    (None, 4, 45)        12420       ['embedding[0][0]']              \n",
            "\n",
            " batch_normalization (BatchNorm  (None, 4, 45)       180         ['lstm[0][0]']                   \n",
            "\n",
            " alization)                                                                                       \n",
            "\n",
            " lstm_1 (LSTM)                  (None, 120)          79680       ['batch_normalization[0][0]']    \n",
            "\n",
            " lstm_2 (LSTM)                  (None, 10)           2240        ['batch_normalization[0][0]']    \n",
            "\n",
            " batch_normalization_1 (BatchNo  (None, 120)         480         ['lstm_1[0][0]']                 \n",
            "\n",
            " rmalization)                                                                                     \n",
            "\n",
            " batch_normalization_2 (BatchNo  (None, 10)          40          ['lstm_2[0][0]']                 \n",
            "\n",
            " rmalization)                                                                                     \n",
            "\n",
            " act_output (Dense)             (None, 17)           2057        ['batch_normalization_1[0][0]']  \n",
            "\n",
            " outcome_output (Dense)         (None, 2)            22          ['batch_normalization_2[0][0]']  \n",
            "\n",
            "==================================================================================================\n",
            "\n",
            "Total params: 97,533\n",
            "\n",
            "Trainable params: 97,183\n",
            "\n",
            "Non-trainable params: 350\n",
            "\n",
            "__________________________________________________________________________________________________\n",
            "\n",
            "None\n",
            "  0% 0/20 [00:04<?, ?it/s, best loss: ?]/usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/adam.py:105: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.\n",
            "  super(Adam, self).__init__(name, **kwargs)\n",
            "\n",
            "Epoch 1/500\n",
            "\n",
            "1050/1050 - 29s - loss: 1.9560 - act_output_loss: 1.4836 - outcome_output_loss: 0.4724 - act_output_accuracy: 0.5220 - outcome_output_accuracy: 0.8235 - val_loss: 1.7347 - val_act_output_loss: 1.2903 - val_outcome_output_loss: 0.4445 - val_act_output_accuracy: 0.5514 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 29s/epoch - 28ms/step\n",
            "\n",
            "Epoch 2/500\n",
            "\n",
            "1050/1050 - 16s - loss: 1.6707 - act_output_loss: 1.2207 - outcome_output_loss: 0.4500 - act_output_accuracy: 0.5787 - outcome_output_accuracy: 0.8371 - val_loss: 1.5739 - val_act_output_loss: 1.1367 - val_outcome_output_loss: 0.4372 - val_act_output_accuracy: 0.6176 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 16s/epoch - 16ms/step\n",
            "\n",
            "Epoch 3/500\n",
            "\n",
            "1050/1050 - 17s - loss: 1.5994 - act_output_loss: 1.1520 - outcome_output_loss: 0.4474 - act_output_accuracy: 0.5869 - outcome_output_accuracy: 0.8371 - val_loss: 1.4981 - val_act_output_loss: 1.0622 - val_outcome_output_loss: 0.4358 - val_act_output_accuracy: 0.6110 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 17s/epoch - 16ms/step\n",
            "\n",
            "Epoch 4/500\n",
            "\n",
            "1050/1050 - 16s - loss: 1.5523 - act_output_loss: 1.1073 - outcome_output_loss: 0.4450 - act_output_accuracy: 0.6027 - outcome_output_accuracy: 0.8369 - val_loss: 1.5035 - val_act_output_loss: 1.0683 - val_outcome_output_loss: 0.4352 - val_act_output_accuracy: 0.6300 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 16s/epoch - 16ms/step\n",
            "\n",
            "Epoch 5/500\n",
            "\n",
            "1050/1050 - 16s - loss: 1.5293 - act_output_loss: 1.0854 - outcome_output_loss: 0.4438 - act_output_accuracy: 0.6065 - outcome_output_accuracy: 0.8371 - val_loss: 1.5464 - val_act_output_loss: 1.1116 - val_outcome_output_loss: 0.4348 - val_act_output_accuracy: 0.6086 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 16s/epoch - 15ms/step\n",
            "\n",
            "Epoch 6/500\n",
            "\n",
            "1050/1050 - 17s - loss: 1.5030 - act_output_loss: 1.0594 - outcome_output_loss: 0.4435 - act_output_accuracy: 0.6140 - outcome_output_accuracy: 0.8371 - val_loss: 1.5060 - val_act_output_loss: 1.0717 - val_outcome_output_loss: 0.4343 - val_act_output_accuracy: 0.6090 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 17s/epoch - 17ms/step\n",
            "\n",
            "Epoch 7/500\n",
            "\n",
            "1050/1050 - 16s - loss: 1.4894 - act_output_loss: 1.0472 - outcome_output_loss: 0.4422 - act_output_accuracy: 0.6127 - outcome_output_accuracy: 0.8371 - val_loss: 1.4981 - val_act_output_loss: 1.0625 - val_outcome_output_loss: 0.4356 - val_act_output_accuracy: 0.6086 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 16s/epoch - 16ms/step\n",
            "\n",
            "Epoch 8/500\n",
            "\n",
            "1050/1050 - 16s - loss: 1.4904 - act_output_loss: 1.0483 - outcome_output_loss: 0.4422 - act_output_accuracy: 0.6129 - outcome_output_accuracy: 0.8371 - val_loss: 1.4551 - val_act_output_loss: 1.0219 - val_outcome_output_loss: 0.4332 - val_act_output_accuracy: 0.6229 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 16s/epoch - 16ms/step\n",
            "\n",
            "Epoch 9/500\n",
            "\n",
            "1050/1050 - 16s - loss: 1.4754 - act_output_loss: 1.0338 - outcome_output_loss: 0.4416 - act_output_accuracy: 0.6168 - outcome_output_accuracy: 0.8371 - val_loss: 1.4810 - val_act_output_loss: 1.0461 - val_outcome_output_loss: 0.4349 - val_act_output_accuracy: 0.6176 - val_outcome_output_accuracy: 0.8414 - lr: 0.0016 - 16s/epoch - 16ms/step\n",
            "\n",
            "Epoch 10/500\n",
            "\n",
            "  0% 0/20 [02:46<?, ?it/s, best loss: ?]"
          ]
        }
      ],
      "source": [
        "!python main.py"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "Untitled0.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyP9cNYIpWXAjp6fIodgatCD"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}