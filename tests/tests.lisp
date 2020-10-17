;;; This library implements a simple neural network.
;;; Copyright 2019-2020 Guillaume Le Vaillant
;;; This library is free software released under the GNU GPL-3 license.

(defpackage :simple-neural-network/test
  (:use :cl :fiveam :simple-neural-network))

(in-package :simple-neural-network/test)


(def-suite snn-tests
  :description "Unit tests for simple neural network.")

(in-suite snn-tests)


(test nn-xor
  (let ((inputs '(#(0.0d0 0.0d0)
                  #(0.0d0 1.0d0)
                  #(1.0d0 0.0d0)
                  #(1.0d0 1.0d0)))
        (targets '(#(0.0d0)
                   #(1.0d0)
                   #(1.0d0)
                   #(0.0d0)))
        (nn (create-neural-network 2 1 0.5d0 2)))
    (dotimes (i 30000)
      (train nn inputs targets))
    (is (> 0.2 (aref (predict nn #(0.0d0 0.0d0)) 0)))
    (is (< 0.8 (aref (predict nn #(1.0d0 0.0d0)) 0)))
    (is (< 0.8 (aref (predict nn #(0.0d0 1.0d0)) 0)))
    (is (> 0.2 (aref (predict nn #(1.0d0 1.0d0)) 0)))))

(defun mnist-file-path (filename)
  (asdf:system-relative-pathname "simple-neural-network"
                                 (concatenate 'string "tests/" filename)))

(defun mnist-read-and-normalize-image (data offset)
  (let ((image (make-array (* 28 28)
                           :element-type 'double-float
                           :initial-element 0.0d0)))
    (dotimes (i (* 28 28) image)
      (setf (aref image i) (/ (aref data (+ offset i)) 255.0d0)))))

(defun mnist-read-and-normalize-label (data offset)
  (let ((target (make-array 10
                            :element-type 'double-float
                            :initial-element 0.0d0))
        (category (aref data offset)))
    (setf (aref target category) 1.0d0)
    target))

(defun mnist-load (type)
  (destructuring-bind (n-images images-path labels-path)
      (if (eql type :train)
          (list 60000
                (mnist-file-path "train-images-idx3-ubyte.gz")
                (mnist-file-path "train-labels-idx1-ubyte.gz"))
          (list 10000
                (mnist-file-path "t10k-images-idx3-ubyte.gz")
                (mnist-file-path "t10k-labels-idx1-ubyte.gz")))
    (let ((images (with-open-file (f images-path
                                     :element-type '(unsigned-byte 8))
                    (let ((images-data (chipz:decompress nil 'chipz:gzip f)))
                      (loop for i from 0 below n-images
                            for offset = (+ 16 (* i 28 28))
                            collect (mnist-read-and-normalize-image images-data
                                                                    offset)))))
          (labels (with-open-file (f labels-path
                                     :element-type '(unsigned-byte 8))
                    (let ((labels-data (chipz:decompress nil 'chipz:gzip f)))
                      (loop for i from 0 below n-images
                            collect (mnist-read-and-normalize-label labels-data
                                                                    (+ 8 i)))))))
      (values images labels))))

(test nn-mnist
  (let ((nn (create-neural-network (* 28 28) 10 0.5d0 128)))
    (multiple-value-bind (inputs targets) (mnist-load :train)
      (train nn inputs targets))
    (multiple-value-bind (inputs targets) (mnist-load :test)
      (is (< 0.8 (accuracy nn inputs targets))))))
