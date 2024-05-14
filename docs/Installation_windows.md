
# Windows Build Instructions

This document outlines the installation instructions necessary to set up the project environment to **run the examples**. We offer guidelines for using Docker or Anaconda as preferred environment management tools. Additionally, a `requirements.txt` file is provided, allowing you the flexibility to utilize any environment manager that suits your needs best.

Please follow the links below to navigate to the specific instructions for each environment setup option:

- [Anaconda Setup Instructions](#anaconda-setup-instructions)
- [Docker Setup Instructions](#docker-setup-instructions)

## Anaconda

### Installation

1. Install [Ananconda](https://docs.anaconda.com/free/anaconda/install/linux/).

2. Open Anaconda and create a new environment called  **Localization** with python **3.11**.

4. Activate the new environment in **conda command propmpt**:

    <code>conda activate localization</code>

5. Navigate to the root of this project.

   <code>cd  "path_to_project/" </code>

6. Install the requirements:

    <code>pip install -r requirements.txt</code>

### Project Set Up

1. **Code editor:** 
   - Open the code editor of your preference in the root folder of this project. (We recommend using vscode)

2. **Activate environment** 
   - Activate your environment in the root folder of the project.

      <code>conda activate localization</code>

3. **(Optional) run jupyter:** 
   - If your code editor doesn't allow to open jupyter files, you can open jupyter or jupter-lab in Anaconda an navigate to the root folder of this project. For this you will need to install **Jupyter** or **Jupyter-Lab** in Anacoda Navigator and Launch it.

4. **Begin Development:**
   - Begin your development. You can create Jupyter notebooks or scripts to start developing your project with your preferred code editor. Ensure that you are running the code within the created environment.

## Docker
### Installation

1. **Visual Studio Code:** Install Visual Studio Code from the [official site](https://code.visualstudio.com/).
2. **VSCode Extensions:** Install the Dev Container extension within Visual Studio Code.
3. **Windows Subsystem for Linux (WSL):** Ensure that you have WSL installed and updated to WSL 2. [Learn how to install/upgrade to WSL 2](https://docs.microsoft.com/en-us/windows/wsl/install).
4. **Vcxsrv:** Download and install Vcxsrv from [SourceForge](https://sourceforge.net/projects/vcxsrv/).
5. **Docker:** Install Docker for Windows from the [Docker website](https://www.docker.com/products/docker-desktop).
### Project Setup

1. **Start Vcxsrv:** Launch the Vcxsrv application (xlaunch program) on your system.
   - Select **Multiple windows** and click **Next**.
   - Choose **Start no client** and click **Next**.
   - Check **Disable access control** and finish the setup.

     ![Disable Access Control](https://user-images.githubusercontent.com/27258035/225386074-df1976a5-6257-4533-997e-d6d770f1335b.png)

2. **Prepare Visual Studio Code:**
   - Open Visual Studio Code.
   - Open your project folder using the shortcut `Ctrl+K Ctrl+O` (make sure you select the root directory of the project).
   - Navigate to `.devcontainer/devcontainers` in the project directory and find the appropriate JSON configuration file for your setup (`devcontainer_windows.json`). Copy its contents to `.devcontainer/devcontainer.json`.

3. **Reopen in Container:**
   - Open the Command Palette in VSCode (lower left blue button).

     ![VSCode Command Palette](https://github.com/dimatura/pypcd/assets/27258035/62e7d831-f1d4-4aa6-ac2e-8982e1a37954)

   - Choose **Reopen in Container** to restart VSCode in the container environment.

     ![Reopen in Container](https://github.com/dimatura/pypcd/assets/27258035/102f3984-2ab8-4c96-87cc-6a6315ed3f00)

4. **Verify X Display Forwarding:**
   - Open a new terminal in VSCode and type `xclock` to ensure the display is being forwarded correctly.
   - If the terminal shows: `bash: xclock: command not found`. Install `xclock` by using:
    `sudo apt install x11-apps`

5. **Begin Development:**
   - Begin your development. You can create Jupyter notebooks or scripts to start developing your project with your preferred code editor. Ensure that you are running the code within the created environment.
