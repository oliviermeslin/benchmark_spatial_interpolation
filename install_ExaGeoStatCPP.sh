
# Run in R
# install.packages(c("Rcpp", "assertthat"))

############
# Install dependencies of ExaGeoStatCPP
############

# Install libtool
sudo apt update
sudo apt install libtool libtool-bin

# Install hwloc
sudo apt install hwloc libhwloc-dev

# Install starpu
# sudo apt install libstarpu-dev libstarpu-1.4-4t64 starpu-tools
sudo apt install libstarpu-1.4-4t64 libstarpu-dev starpu-tools starpu-examples

# Install chameleon
wget https://gitlab.inria.fr/api/v4/projects/616/packages/generic/ubuntu_22.04/1.2.0/chameleon_1.2.0-1_amd64.deb
sudo apt install ./chameleon_1.2.0-1_amd64.deb

# Install LAPACKE
# sudo apt install liblapacke-dev liblapack-dev libblas-dev libopenblas-dev
sudo apt install liblapacke64-dev

############
# Install ExaGeoStatCPP
############
# Clone the ExaGeoStatCPP repo
cd /home/onyxia/work/
git clone https://github.com/ecrc/ExaGeoStatCPP.git 

cd ExaGeoStatCPP
R CMD INSTALL . --configure-args="-r"



