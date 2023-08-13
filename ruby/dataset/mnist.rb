require "active_support/all"
require "digest/md5"
require "faraday"

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

  def load
    ensure_files_loaded
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
