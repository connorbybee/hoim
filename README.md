 
## Code for Efficient Optimization with Higher-Order Ising Machines
This repo contains the code used to produce the results in Efficient Optimization with Higher-Order Ising Machines [1].

[1] Bybee, C., Kleyko, D., Nikonov, D. E., Khosrowshahi, A., Olshausen, B. A., & Sommer, F. T. (2022). Efficient Optimization with Higher-Order Ising Machines. arXiv preprint arXiv:2212.03426.

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

* Clone the repo
   ```sh
   git clone https://github.com/connorbybee/hoim.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Dataset

The following steps download the dataset used in this paper. 

```sh
mkdir sat
cd sat
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf50-218.tar.gz
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf100-430.tar.gz
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz
tar xvf *.gz
cd ..
```


<!-- USAGE EXAMPLES -->
## Simulation

Two examples scripts are provided to run the higher-order and second-order oscillator ising machines.

```sh
python sat_3rd.py
python sat_2nd.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Analysis
An example notebook is provided to unpickle the results from simulations and compute the energy and fraction of satisfied problem isntances.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Connor Bybee

Project Link: [https://github.com/connorbybee/hoim.git](https://github.com/connorbybee/hoim.git)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
