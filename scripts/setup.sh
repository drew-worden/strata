#!/bin/bash

# Install Poetry
install_poetry() {
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo "Poetry installed successfully."
}

# Open NVIDIA GPU process manager
open_nvidia_smi() {
    echo "Opening NVIDIA GPU process manager..."
    watch -n 0.1 nvidia-smi
}

# Update and upgrade the system
update_system() {
    echo "Updating and upgrading the system..."
    sudo apt-get update && sudo apt-get upgrade -y
    echo "System updated and upgraded successfully."
}

# Install common dependencies
install_dependencies() {
    echo "Installing common dependencies..."
    sudo apt-get install -y build-essential curl python3 python3-pip
    echo "Dependencies installed successfully."
}

# Install project packages
install_poetry_packages() {
    echo "Installing project packages..."
    poetry add package1 package2 package3
    echo "Project packages installed successfully."
}

# Main script execution
main() {
    update_system
    install_dependencies
    install_poetry
    install_poetry_packages
    open_nvidia_smi
}

# Run the main function
main