
# Linux Build Instructions

This document outlines the installation instructions necessary to set up the project environment to **run the examples**. We offer guidelines for using Docker or Anaconda as preferred environment management tools. Additionally, a `requirements.txt` file is provided, allowing you the flexibility to utilize any environment manager that suits your needs best.

Please follow the links below to navigate to the specific instructions for each environment setup option:

- [Anaconda Setup Instructions](#anaconda-setup-instructions)
- [Docker Setup Instructions](#docker-setup-instructions)

## Anaconda

### Installation

1. Install [Ananconda](https://docs.anaconda.com/free/anaconda/install/linux/).

2. Initialize Anaconda by opening a terminal and running:

   <code>conda init</code>

3. Create a new environment named **Localization** with python **3.11**.

    <code>conda create -n localization python=3.11</code>

4. Activate the new environment:

    <code>conda activate localization</code>

5. Navigate to the root of this project.

   <code>cd  "path_to_project/" </code>

4. Install the requirements:

    <code>pip install -r requirements.txt</code>

### Project Set Up

1. **Code editor:** 
   - Open the code editor of your preference in the root folder of this project. (We recommend using vscode)

2. **Activate environment** 
   - Activate your environment in the root folder of the project.

      <code>conda activate localization</code>

3. **(Optional) run jupyter:** 
   - If your code editor doesn't support opening Jupyter notebook files, you can open Jupyter in the root folder by running:

      <code>pip install jupyter</code>
   
      and then:

      <code>jupyter notebook .</code>

4. **Begin Development:**
   - Begin your development. You can create Jupyter notebooks or scripts to start developing your project with your preferred code editor. Ensure that you are running the code within the created environment.


## Docker

### Installation

1. **Docker:** Install Docker Engine for linux from the [docker website](https://docs.docker.com/engine/install/ubuntu/).
2. **Visual Studio Code:** Install Visual Studio Code from the [official site](https://code.visualstudio.com/).
3. **VSCode Extensions:** Install the **Remote SSH** and **Dev Container** extension within Visual Studio Code.

### Project Set Up

1. **Prepare Visual Studio Code:**
   - Open Visual Studio Code.
   - Open your project folder using the shortcut `Ctrl+K Ctrl+O` (make sure you select the root directory of the project).
   - Navigate to `.devcontainer/devcontainers` in the project directory and find the appropriate JSON configuration file for your setup (`devcontainer_linux.json`). Copy its contents to `.devcontainer/devcontainer.json`.

2. **Reopen in Container:**
   - Open the Command Palette in VSCode (lower left blue button).

     ![VSCode Command Palette](https://github.com/dimatura/pypcd/assets/27258035/62e7d831-f1d4-4aa6-ac2e-8982e1a37954)

   - Choose **Reopen in Container** to restart VSCode in the container environment.

     ![Reopen in Container](https://github.com/dimatura/pypcd/assets/27258035/102f3984-2ab8-4c96-87cc-6a6315ed3f00)

3. **Verify X Display Forwarding:**
   - Open a new terminal in VSCode and type `xclock` to ensure the display is being forwarded correctly.
   - If the terminal shows: `bash: xclock: command not found`. Install `xclock` by using:
    `sudo apt install x11-apps`

4. **Begin Development:**
   - Begin your development. You can create Jupyter notebooks or scripts to start developing your project with your preferred code editor. Ensure that you are running the code within the created environment.
