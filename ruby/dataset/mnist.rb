require "active_support/all"
require "digest/md5"
require "faraday"
require "torch-rb"

module Mnist
  extend self

  BASE_URL = "http://yann.lecun.com/exdb/mnist/".freeze

  FILE_NAMES = {
    train_img: "train-images-idx3-ubyte.gz",
    train_label: "train-labels-idx1-ubyte.gz",
    test_img: "t10k-images-idx3-ubyte.gz",
    test_label: "t10k-labels-idx1-ubyte.gz"
  }.with_indifferent_access.freeze

  FILE_CHECKSUMS = {
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432"
  }.with_indifferent_access.freeze

  def load(_flatten: true, _normalize: false)
    ensure_files_loaded

    x_train = load_train_images
    t_train = load_train_labels
    x_test = load_test_images
    t_test = load_test_labels

    { x_train:, t_train:, x_test:, t_test: }
  end

  def load_train_images
    Zlib::GzipReader.open(File.expand_path(FILE_NAMES[:train_img], __dir__)) do |f|
      magic, n_images = f.read(8).unpack("N2")
      raise "Invalid MNIST image file" if magic != 2051

      n_rows, n_cols = f.read(8).unpack("N2")
      n_images.times
        .map { f.read(n_rows * n_cols) }
        .map { |img| img.unpack("C*") }
        .then { |images| Torch.tensor(images) }
    end
  end

  def load_train_labels
    Zlib::GzipReader.open(File.expand_path(FILE_NAMES[:train_label], __dir__)) do |f|
      magic, n_labels = f.read(8).unpack("N2")
      raise "Invalid MNIST label file" if magic != 2049

      labels = f.read(n_labels).unpack("C*")
      Torch.tensor(labels)
    end
  end

  def load_test_images
    Zlib::GzipReader.open(File.expand_path(FILE_NAMES[:test_img], __dir__)) do |f|
      magic, n_images = f.read(8).unpack("N2")
      raise "Invalid MNIST image file" if magic != 2051

      n_rows, n_cols = f.read(8).unpack("N2")
      n_images.times
        .map { f.read(n_rows * n_cols) }
        .map { |img| img.unpack("C*") }
        .then { |images| Torch.tensor(images) }
    end
  end

  def load_test_labels
    Zlib::GzipReader.open(File.expand_path(FILE_NAMES[:test_label], __dir__)) do |f|
      magic, n_labels = f.read(8).unpack("N2")
      raise "Invalid MNIST label file" if magic != 2049

      labels = f.read(n_labels).unpack("C*")
      Torch.tensor(labels)
    end
  end

  private

    def ensure_files_loaded
      FILE_NAMES.each do |_type, file_name|
        local_path = File.expand_path(file_name, __dir__)
        next if check_file_md5(local_path)

        Faraday.get("#{BASE_URL}/#{file_name}")
          .body
          .then { |blob| File.write(local_path, blob) }
      end
    end

    def check_file_md5(local_path)
      if File.exist?(local_path) && (Digest::MD5.file(local_path) == FILE_CHECKSUMS[File.basename(local_path)])
        true
      elsif File.exist?(local_path)
        File.delete(local_path)
        false
      end
    end
end
