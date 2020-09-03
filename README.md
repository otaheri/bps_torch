# bps_torch
A Pytorch implementation of the [bps](https://github.com/sergeyprokudin/bps) representation using chamfer distance on GPU. This implementation is very fast and was used for the [GrabNet](https://github.com/otaheri/GrabNet) model.

**Basis Point Set (BPS)** is a simple and efficient method for encoding 3D point clouds into fixed-length representations. For the original implementation please visit [this implementation](https://github.com/amzn/basis-point-sets) by [Sergey Prokudin](https://github.com/sergeyprokudin).


### Requirements

- Python >= 3.7
- PyTorch >= 1.1.0 
- Numpy >= 1.16.2
- [chamfer_distance](https://github.com/otaheri/chamfer_distance)

### Installation


```
pip install git+https://github.com/otaheri/bps_torch
```

### Demos

#Coming Soon ...


## Citation

If you use this code in your research, please consider citing:
```
@inproceedings{prokudin2019efficient,
  title={Efficient Learning on Point Clouds With Basis Point Sets},
  author={Prokudin, Sergey and Lassner, Christoph and Romero, Javier},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4332--4341},
  year={2019}
}
```

## License

This library is licensed under the MIT-0 License of the original implementation. See the [LICENSE](https://github.com/sergeyprokudin/bps/blob/master/LICENSE) file.

## Contact
The code of this repository was implemented by [Omid Taheri](https://ps.is.tue.mpg.de/person/otaheri).

For questions, please contact [omid.taheri@tue.mpg.de](mailto:omid.taheri@tue.mpg.de).
