;;; This library implements a simple neural network.
;;; Copyright 2019-2020 Guillaume Le Vaillant
;;; This library is free software released under the GNU GPL-3 license.

(defpackage :simple-neural-network
  (:nicknames :snn)
  (:use :cl)
  (:export #:create-neural-network
           #:train
           #:predict
           #:store
           #:restore
           #:index-of-max-value
           #:same-category-p
           #:accuracy))

(in-package :simple-neural-network)


(defstruct neural-network
  layers
  weights
  biases
  deltas)

(declaim (inline activation))
(defun activation (x)
  "Sigmoid activation function for the neurons."
  (declare (type double-float x))
  ;; Use some special cases to prevent possible floating point overflows when
  ;; computing the exponential.
  (cond
    ((> x 100.0d0)
     1.0d0)
    ((< x -100.0d0)
     0.0d0)
    (t
     (/ 1.0d0 (+ 1.0d0 (exp (- x)))))))

(declaim (inline activation-prime))
(defun activation-prime (x)
  "Derivative of the activation function."
  (declare (type double-float x))
  (* x (- 1.0d0 x)))

(defun make-random-weights (input-size output-size)
  "Generate a matrix (INPUT-SIZE * OUTPUT-SIZE) of random weights between -1
and 1."
  (let ((weights (make-array (list input-size output-size)
                             :element-type 'double-float
                             :initial-element 0.0d0)))
    (dotimes (i input-size weights)
      (dotimes (j output-size)
        (setf (aref weights i j) (1- (random 2.0d0)))))))

(defun make-random-biases (size)
  "Generate a vector of SIZE random biases between -1 and 1."
  (let ((biases (make-array size
                            :element-type 'double-float
                            :initial-element 0.0d0)))
    (dotimes (i size biases)
      (setf (aref biases i) (1- (random 2.0d0))))))

(defun create-neural-network (input-size output-size &rest hidden-layers-sizes)
  "Create a neural network having INPUT-SIZE inputs, OUTPUT-SIZE outputs, and
optionally some intermediary layers whose sizes are specified by
HIDDEN-LAYERS-SIZES. The neural network is initialized with random weights and
biases."
  (let* ((*random-state* (make-random-state t))
         (layer-sizes (append (list input-size)
                              hidden-layers-sizes
                              (list output-size)))
         (layers (mapcar (lambda (size)
                           (make-array size
                                       :element-type 'double-float
                                       :initial-element 0.0d0))
                         layer-sizes))
         (weights (butlast (maplist (lambda (sizes)
                                      (let ((input-size (first sizes))
                                            (output-size (second sizes)))
                                        (when output-size
                                          (make-random-weights input-size
                                                               output-size))))
                                    layer-sizes)))
         (biases (mapcar (lambda (size)
                           (make-random-biases size))
                         (rest layer-sizes)))
         (deltas (mapcar (lambda (size)
                           (make-array size
                                       :element-type 'double-float
                                       :initial-element 0.0d0))
                         (rest layer-sizes))))
    (make-neural-network :layers layers
                         :weights weights
                         :biases biases
                         :deltas deltas)))

(defun set-input (neural-network input)
  "Set the input layer of the NEURAL-NETWORK to INPUT."
  (declare (type vector input))
  (let ((input-layer (first (neural-network-layers neural-network))))
    (declare (type (simple-array double-float (*)) input-layer))
    (dotimes (i (length input-layer) neural-network)
      (setf (aref input-layer i) (aref input i)))))

(defun get-output (neural-network)
  "Return the output layer of the NEURAL-NETWORK."
  (first (last (neural-network-layers neural-network))))

(defun compute-values (input output weights biases)
  "Compute the values of the neurons in the OUTPUT layer."
  (declare (type (simple-array double-float (*)) input output biases)
           (type (simple-array double-float (* *)) weights)
           (optimize (speed 3)))
  (dotimes (i (length output))
    (let ((aggregation (aref biases i)))
      (declare (type double-float aggregation))
      (dotimes (j (length input))
        (incf aggregation (* (aref input j) (aref weights j i))))
      (setf (aref output i) (activation aggregation)))))

(defun propagate (neural-network)
  "Propagate the values of the input layer of the NEURAL-NETWORK to the output
layer."
  (do ((layers (neural-network-layers neural-network)
               (rest layers))
       (weights (neural-network-weights neural-network)
                (rest weights))
       (biases (neural-network-biases neural-network)
               (rest biases)))
      ((endp weights) neural-network)
    (compute-values (first layers)
                    (second layers)
                    (first weights)
                    (first biases))))

(defun compute-output-delta (neural-network target)
  "Compute the error between the output layer of the NEURAL-NETWORK and the
TARGET."
  (declare (type vector target))
  (let ((output (get-output neural-network))
        (delta (first (last (neural-network-deltas neural-network)))))
    (declare (type (simple-array double-float (*)) output delta))
    (dotimes (i (length output) delta)
      (let* ((value (aref output i))
             (diff (- value (aref target i))))
        (declare (type double-float value diff))
        (setf (aref delta i) (* (activation-prime value) diff))))))

(defun compute-delta (previous-delta output weights delta)
  "Compute the error of the OUTPUT layer based on the error of the next layer."
  (declare (type (simple-array double-float (*)) previous-delta output delta)
           (type (simple-array double-float (* *)) weights)
           (optimize (speed 3)))
  (dotimes (i (length delta) delta)
    (let ((value 0.0d0))
      (declare (type double-float value))
      (dotimes (j (length previous-delta))
        (incf value (* (aref previous-delta j) (aref weights i j))))
      (setf (aref delta i) (* (activation-prime (aref output i))
                              value)))))

(defun backpropagate (neural-network)
  "Propagate the error of the output layer of the NEURAL-NETWORK back to the
first layer."
  (do ((layers (rest (reverse (neural-network-layers neural-network)))
               (rest layers))
       (weights (reverse (neural-network-weights neural-network))
                (rest weights))
       (deltas (reverse (neural-network-deltas neural-network))
               (rest deltas)))
      ((endp (rest deltas)) neural-network)
    (compute-delta (first deltas)
                   (first layers)
                   (first weights)
                   (second deltas))))

(defun update-weights (input weights delta learning-rate)
  "Update the WEIGHTS of a layer."
  (declare (type (simple-array double-float (*)) input delta)
           (type (simple-array double-float (* *)) weights)
           (type double-float learning-rate)
           (optimize (speed 3)))
  (dotimes (i (length delta))
    (let ((gradient (* learning-rate (aref delta i))))
      (declare (type double-float gradient))
      (dotimes (j (length input))
        (decf (aref weights j i) (* gradient (aref input j)))))))

(defun update-biases (biases delta learning-rate)
  "Update the BIASES of a layer."
  (declare (type (simple-array double-float (*)) biases delta)
           (type double-float learning-rate))
  (dotimes (i (length biases))
    (decf (aref biases i) (* learning-rate (aref delta i)))))

(defun update-weights-and-biases (neural-network learning-rate)
  "Update all the weights and biases of the NEURAL-NETWORK."
  (mapc (lambda (layer weights biases delta)
          (update-weights layer weights delta learning-rate)
          (update-biases biases delta learning-rate))
        (neural-network-layers neural-network)
        (neural-network-weights neural-network)
        (neural-network-biases neural-network)
        (neural-network-deltas neural-network))
  neural-network)

(defun train (neural-network inputs targets learning-rate)
  "Train the NEURAL-NETWORK at a given LEARNING-RATE using some INPUTS and
TARGETS."
  (let ((learning-rate (coerce learning-rate 'double-float)))
    (mapc (lambda (input target)
            (set-input neural-network input)
            (propagate neural-network)
            (compute-output-delta neural-network target)
            (backpropagate neural-network)
            (update-weights-and-biases neural-network learning-rate))
          inputs
          targets)
    neural-network))

(defun predict (neural-network input &optional output)
  "Return the output computed by the NEURAL-NETWORK for a given INPUT. If
OUTPUT is not NIL, the output is written in it, otherwise a new vector is
allocated."
  (set-input neural-network input)
  (propagate neural-network)
  (if output
      (replace output (get-output neural-network))
      (copy-seq (get-output neural-network))))

(defun store (neural-network place)
  "Store the NEURAL-NETWORK to PLACE, which must be a stream or
a pathname-designator."
  ;; cl-store only supports serialization of structures on SBCL and CMUCL.
  ;; Use a list instead to support other implementations.
  (cl-store:store (list (neural-network-layers neural-network)
                        (neural-network-weights neural-network)
                        (neural-network-biases neural-network)
                        (neural-network-deltas neural-network))
                  place))

(defun restore (place)
  "Restore the neural network stored in PLACE, which must be a stream or
a pathname-designator."
  (destructuring-bind (layers weights biases deltas) (cl-store:restore place)
    (make-neural-network :layers layers
                         :weights weights
                         :biases biases
                         :deltas deltas)))

(defun index-of-max-value (values)
  "Return the index of the greatest value in VALUES."
  (do ((size (length values))
       (index 0)
       (maximum (aref values 0))
       (i 1 (1+ i)))
      ((>= i size) index)
    (when (> (aref values i) maximum)
      (setf index i)
      (setf maximum (aref values i)))))

(defun same-category-p (output target)
  "Return T if calls to INDEX-OF-MAX-VALUE on OUTPUT and TARGET return the same
value, and NIL otherwise."
  (= (index-of-max-value output)
     (index-of-max-value target)))

(defun accuracy (neural-network inputs targets &key (test #'same-category-p))
  "Return the rate of good guesses computed by the NEURAL-NETWORK when testing
it with some INPUTS and TARGETS. TEST must be a function taking an output and
a target returning T if the output is considered to be close enough to the
target, and NIL otherwise. SAME-CATEGORY-P is used by default."
  (let* ((output (copy-seq (get-output neural-network)))
         (guesses (mapcar (lambda (input target)
                            (funcall test
                                     (predict neural-network input output)
                                     target))
                         inputs
                         targets)))
    (/ (count t guesses) (length inputs))))
