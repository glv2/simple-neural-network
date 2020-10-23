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


(deftype double-float-array ()
  '(simple-array double-float (*)))

(defstruct neural-network
  layers
  weights
  biases
  deltas)

(declaim (inline activation))
(defun activation (x)
  "Activation function for the neurons."
  (declare (type double-float x))
  (* 1.7159d0 (tanh (* 2/3 x))))

(declaim (inline activation-prime))
(defun activation-prime (x)
  "Derivative of the activation function."
  (declare (type double-float x))
  (- 1.1439d0 (* 0.38852d0 x x)))

(defun make-random-weights (input-size output-size)
  "Generate a matrix (OUTPUT-SIZE * INPUT-SIZE) of random weights."
  (let ((weights (make-array (* output-size input-size)
                             :element-type 'double-float
                             :initial-element 0.0d0))
        (r (sqrt (/ 6.0d0 (+ input-size output-size)))))
    (dotimes (i (* output-size input-size) weights)
      (setf (aref weights i) (- (random (* 2.0d0 r)) r)))))

(defun make-random-biases (size)
  "Generate a vector of SIZE random biases."
  (let ((biases (make-array size
                            :element-type 'double-float
                            :initial-element 0.0d0))
        (r (/ 1.0d0 (sqrt size))))
    (dotimes (i size biases)
      (setf (aref biases i) (- (random (* 2.0d0 r)) r)))))

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
    (declare (type double-float-array input-layer))
    (replace input-layer input)
    neural-network))

(defun get-output (neural-network)
  "Return the output layer of the NEURAL-NETWORK."
  (first (last (neural-network-layers neural-network))))

(defun compute-values (input output weights biases)
  "Compute the values of the neurons in the OUTPUT layer."
  (declare (type double-float-array input output weights biases)
           (optimize (speed 3) (safety 0)))
  (let ((input-size (length input))
        (output-size (length output)))
    (declare (type fixnum input-size output-size))
    (flet ((compute (i)
             (declare (type fixnum i))
             (let ((offset (the fixnum (* i input-size)))
                   (aggregation (aref biases i)))
               (declare (type fixnum offset)
                        (type double-float aggregation))
               (dotimes (j input-size)
                 (declare (type fixnum j))
                 (incf aggregation (* (aref input j)
                                      (aref weights (+ offset j)))))
               (setf (aref output i) (activation aggregation)))))
      (if lparallel:*kernel*
          (lparallel:pdotimes (i output-size)
            (compute i))
          (dotimes (i output-size)
            (compute i))))))

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
    (declare (type double-float-array output delta))
    (dotimes (i (length output) delta)
      (let* ((value (aref output i))
             (diff (- value (aref target i))))
        (declare (type double-float value diff))
        (setf (aref delta i) (* (activation-prime value) diff))))))

(defun compute-delta (previous-delta output weights delta)
  "Compute the error of the OUTPUT layer based on the error of the next layer."
  (declare (type double-float-array previous-delta output weights delta)
           (optimize (speed 3) (safety 0)))
  (let ((previous-delta-size (length previous-delta))
        (delta-size (length delta)))
    (declare (type fixnum previous-delta-size delta-size))
    (flet ((compute (i)
             (declare (type fixnum i))
             (let ((value 0.0d0))
               (declare (type double-float value))
               (dotimes (j previous-delta-size)
                 (declare (type fixnum j))
                 (let ((offset (+ (the fixnum (* j delta-size)) i)))
                   (declare (type fixnum offset))
                   (incf value (* (aref previous-delta j)
                                  (aref weights offset)))))
               (setf (aref delta i) (* (activation-prime (aref output i))
                                       value)))))
      (if lparallel:*kernel*
          (lparallel:pdotimes (i delta-size delta)
            (compute i))
          (dotimes (i delta-size delta)
            (compute i))))))

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
  (declare (type double-float-array input weights delta)
           (type double-float learning-rate)
           (optimize (speed 3) (safety 0)))
  (let ((input-size (length input))
        (delta-size (length delta)))
    (declare (type fixnum input-size delta-size))
    (flet ((compute (i)
             (declare (type fixnum i))
             (let ((offset (the fixnum (* i input-size)))
                   (gradient (* learning-rate (aref delta i))))
               (declare (type fixnum offset)
                        (type double-float gradient))
               (dotimes (j input-size)
                 (declare (type fixnum j))
                 (decf (aref weights (+ offset j)) (* (aref input j)
                                                      gradient))))))
      (if lparallel:*kernel*
          (lparallel:pdotimes (i delta-size)
            (compute i))
          (dotimes (i delta-size)
            (compute i))))))

(defun update-biases (biases delta learning-rate)
  "Update the BIASES of a layer."
  (declare (type double-float-array biases delta)
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
