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
        (nn (create-neural-network 2 1 2)))
    (dotimes (i 30000)
      (train nn inputs targets))
    (is (> 0.2 (aref (predict nn #(0.0d0 0.0d0)) 0)))
    (is (< 0.8 (aref (predict nn #(1.0d0 0.0d0)) 0)))
    (is (< 0.8 (aref (predict nn #(0.0d0 1.0d0)) 0)))
    (is (> 0.2 (aref (predict nn #(1.0d0 1.0d0)) 0)))))

(defun mnist-file-path (filename)
  (asdf:system-relative-pathname "simple-neural-network"
                                 (concatenate 'string "tests/" filename)))

(defun load-mnist (type)
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
                            collect (coerce (loop with offset = (+ 16 (* i 28 28))
                                                  for j from 0 below (* 28 28)
                                                  collect (aref images-data (+ offset j)))
                                            'vector)))))
          (labels (with-open-file (f labels-path :element-type '(unsigned-byte 8))
                    (let ((labels-data (chipz:decompress nil 'chipz:gzip f)))
                      (loop for i from 0 below n-images
                            collect (let ((v (aref labels-data (+ 8 i)))
                                          (a (make-array 10 :initial-element 0.0d0)))
                                      (setf (aref a v) 1.0d0)
                                      a))))))
      (trivial-garbage:gc :full t)
      (values images labels))))

(test nn-mnist
  (if #+sbcl (< (sb-ext:dynamic-space-size) (expt 2 32))
      #-sbcl nil
      (skip "Dynamic space size too small.")
      (let ((nn (create-neural-network (* 28 28) 10 128)))
        (multiple-value-bind (inputs targets) (load-mnist :train)
          (mapc (lambda (input)
                  (map-into input (lambda (x) (/ x 255.0d0)) input))
                inputs)
          (train nn inputs targets))
        (trivial-garbage:gc :full t)
        (multiple-value-bind (inputs targets) (load-mnist :test)
          (mapc (lambda (input)
                  (map-into input (lambda (x) (/ x 255.0d0)) input))
                inputs)
          (is (< 0.8 (accuracy nn inputs targets)))))))
