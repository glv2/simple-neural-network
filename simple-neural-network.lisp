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
           #:copy
           #:index-of-max-value
           #:same-category-p
           #:accuracy
           #:mean-absolute-error
           #:find-normalization
           #:find-learning-rate))

(in-package :simple-neural-network)


(deftype double-float-array ()
  '(simple-array double-float (*)))

(defstruct neural-network
  layers
  weights
  biases
  gradients
  deltas)

(defun make-double-float-array (size)
  "Make a new array of SIZE double floats."
  (make-array size :element-type 'double-float :initial-element 0.0d0))

(defmacro %dotimes ((var count &optional result) &body body)
  `(if lparallel:*kernel*
       (lparallel:pdotimes (,var ,count ,result)
         ,@body)
       (dotimes (,var ,count ,result)
         ,@body)))

(defmacro %mapc (function list &rest more-lists)
  `(if lparallel:*kernel*
       (lparallel:pmapc ,function ,list ,@more-lists)
       (mapc ,function ,list ,@more-lists)))

(declaim (inline activation))
(defun activation (x)
  "Activation function for the neurons."
  (declare (type double-float x))
  (* 1.7159d0 (tanh (* 0.66667d0 x))))

(declaim (inline activation-prime))
(defun activation-prime (y)
  "Derivative of the activation function."
  (declare (type double-float y))
  ;; Here Y is the result of (ACTIVATION X).
  (- 1.1439d0 (* 0.38852d0 y y)))

(defun make-random-weights (input-size output-size)
  "Generate a matrix (OUTPUT-SIZE * INPUT-SIZE) of random weights."
  (let ((weights (make-double-float-array (* output-size input-size)))
        (r (sqrt (/ 6.0d0 (+ input-size output-size)))))
    (dotimes (i (* output-size input-size) weights)
      (setf (aref weights i) (- (random (* 2.0d0 r)) r)))))

(defun make-random-biases (size)
  "Generate a vector of SIZE random biases."
  (let ((biases (make-double-float-array size))
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
         (layers (mapcar #'make-double-float-array layer-sizes))
         (weights (butlast (maplist (lambda (sizes)
                                      (let ((input-size (first sizes))
                                            (output-size (second sizes)))
                                        (when output-size
                                          (make-random-weights input-size
                                                               output-size))))
                                    layer-sizes)))
         (biases (mapcar #'make-random-biases (rest layer-sizes)))
         (gradients (mapcar (lambda (weights)
                              (make-double-float-array (length weights)))
                            weights))
         (deltas (mapcar #'make-double-float-array (rest layer-sizes))))
    (make-neural-network :layers layers
                         :weights weights
                         :biases biases
                         :gradients gradients
                         :deltas deltas)))

(defun set-input (neural-network input)
  "Set the input layer of the NEURAL-NETWORK to INPUT."
  (declare (type vector input))
  (let ((input-layer (first (neural-network-layers neural-network))))
    (declare (type double-float-array input-layer))
    (replace input-layer input)
    (values)))

(defun get-output (neural-network)
  "Return the output layer of the NEURAL-NETWORK."
  (first (last (neural-network-layers neural-network))))

(declaim (inline compute-value))
(defun compute-value (input output weights biases index)
  "Compute the values of the neuron at INDEX in the OUTPUT layer."
  (declare (type double-float-array input output weights biases)
           (type fixnum index)
           (optimize (speed 3) (safety 0)))
  (let* ((input-size (length input))
         (aggregation (aref biases index)))
    (declare (type fixnum input-size)
             (type double-float aggregation))
    (do ((j 0 (1+ j))
         (k (the fixnum (* index input-size)) (1+ k)))
        ((>= j input-size))
      (declare (type fixnum j k))
      (incf aggregation (* (aref input j) (aref weights k))))
    (setf (aref output index) (activation aggregation))
    (values)))

(defun compute-values (input output weights biases)
  "Compute the values of the neurons in the OUTPUT layer."
  (declare (type double-float-array input output weights biases)
           (optimize (speed 3) (safety 0)))
  (let ((output-size (length output)))
    (declare (type fixnum output-size))
    (%dotimes (i output-size)
      (declare (type fixnum i))
      (compute-value input output weights biases i))))

(defun propagate (neural-network)
  "Propagate the values of the input layer of the NEURAL-NETWORK to the output
layer."
  (do ((layers (neural-network-layers neural-network)
               (rest layers))
       (weights (neural-network-weights neural-network)
                (rest weights))
       (biases (neural-network-biases neural-network)
               (rest biases)))
      ((endp weights))
    (compute-values (first layers)
                    (second layers)
                    (first weights)
                    (first biases)))
  (values))

(defun compute-output-delta (neural-network target)
  "Compute the error between the output layer of the NEURAL-NETWORK and the
TARGET."
  (declare (type vector target))
  (let ((output (get-output neural-network))
        (delta (first (last (neural-network-deltas neural-network)))))
    (declare (type double-float-array output delta))
    (dotimes (i (length output))
      (let* ((value (aref output i))
             (diff (- value (aref target i))))
        (declare (type double-float value diff))
        (setf (aref delta i) (* (activation-prime value) diff))))))

(declaim (inline compute-single-delta))
(defun compute-single-delta (previous-delta output weights delta index)
  "Compute the delta for the neuron at INDEX in the OUTPUT layer."
  (declare (type double-float-array previous-delta output weights delta)
           (type fixnum index)
           (optimize (speed 3) (safety 0)))
  (let ((previous-delta-size (length previous-delta))
        (delta-size (length delta))
        (value 0.0d0))
    (declare (type fixnum previous-delta-size delta-size)
             (type double-float value))
    (dotimes (j previous-delta-size)
      (declare (type fixnum j))
      (let ((offset (+ (the fixnum (* j delta-size)) index)))
        (declare (type fixnum offset))
        (incf value (* (aref previous-delta j) (aref weights offset)))))
    (setf (aref delta index) (* (activation-prime (aref output index)) value))
    (values)))

(defun compute-delta (previous-delta output weights delta)
  "Compute the error of the OUTPUT layer based on the error of the next layer."
  (declare (type double-float-array previous-delta output weights delta)
           (optimize (speed 3) (safety 0)))
  (let ((delta-size (length delta)))
    (declare (type fixnum delta-size))
    (%dotimes (i delta-size delta)
      (declare (type fixnum i))
      (compute-single-delta previous-delta output weights delta i))))

(declaim (inline add-gradient))
(defun add-gradient (input gradients delta index)
  "Add the gradients computed for an input for the weights of the neuron at
INDEX in a layer to the sum of the gradients for previous inputs."
  (declare (type double-float-array input gradients delta)
           (type fixnum index)
           (optimize (speed 3) (safety 0)))
  (let ((input-size (length input))
        (gradient (aref delta index)))
    (declare (type fixnum input-size)
             (type double-float gradient))
    (do ((j (the fixnum (* index input-size)) (1+ j))
         (k 0 (1+ k)))
        ((>= k input-size))
      (declare (type fixnum j k))
      (incf (aref gradients j) (* (aref input k) gradient)))))

(defun add-gradients (input gradients delta)
  "Add the gradients computed for an input to sum of the gradients for previous
inputs."
  (declare (type double-float-array input gradients delta)
           (optimize (speed 3) (safety 0)))
  (let ((delta-size (length delta)))
    (declare (type fixnum delta-size))
    (%dotimes (i delta-size)
      (declare (type fixnum i))
      (add-gradient input gradients delta i))))

(declaim (inline average-gradient))
(defun average-gradient (gradient batch-size)
  "Compute the average gradients for a layer."
  (declare (type double-float-array gradient)
           (type double-float batch-size)
           (optimize (speed 3) (safety 0)))
  (dotimes (i (length gradient))
    (declare (type fixnum i))
    (setf (aref gradient i) (/ (aref gradient i) batch-size))))

(defun average-gradients (neural-network batch-size)
  "Compute the average gradients for the whole NEURAL-NETWORK."
  (let ((batch-size (coerce batch-size 'double-float)))
    (%mapc (lambda (gradient)
             (average-gradient gradient batch-size))
           (neural-network-gradients neural-network))
    (values)))

(defun backpropagate (neural-network)
  "Propagate the error of the output layer of the NEURAL-NETWORK back to the
first layer and compute the gradients."
  (do ((layers (rest (reverse (neural-network-layers neural-network)))
               (rest layers))
       (weights (reverse (neural-network-weights neural-network))
                (rest weights))
       (gradients (reverse (neural-network-gradients neural-network))
                  (rest gradients))
       (deltas (reverse (neural-network-deltas neural-network))
               (rest deltas)))
      ((endp deltas))
    (unless (endp (rest deltas))
      (compute-delta (first deltas)
                     (first layers)
                     (first weights)
                     (second deltas)))
    (add-gradients (first layers)
                   (first gradients)
                   (first deltas)))
  (values))

(defun update-weights (weights gradients learning-rate)
  "Update the WEIGHTS of a layer and clear the GRADIENTS."
  (declare (type double-float-array weights gradients)
           (type double-float learning-rate)
           (optimize (speed 3) (safety 0)))
  (dotimes (i (length weights))
    (declare (type fixnum i))
    (decf (aref weights i) (* learning-rate (aref gradients i)))
    (setf (aref gradients i) 0.0d0)))

(defun update-biases (biases delta learning-rate)
  "Update the BIASES of a layer."
  (declare (type double-float-array biases delta)
           (type double-float learning-rate)
           (optimize (speed 3) (safety 0)))
  (dotimes (i (length biases))
    (decf (aref biases i) (* learning-rate (aref delta i)))))

(defun update-weights-and-biases (neural-network learning-rate)
  "Update all the weights and biases of the NEURAL-NETWORK."
  (%mapc (lambda (weights biases gradients delta)
           (update-weights weights gradients learning-rate)
           (update-biases biases delta learning-rate))
         (neural-network-weights neural-network)
         (neural-network-biases neural-network)
         (neural-network-gradients neural-network)
         (neural-network-deltas neural-network))
  (values))

(defun train (neural-network inputs targets learning-rate
              &optional (batch-size 1))
  "Train the NEURAL-NETWORK at a given LEARNING-RATE using some INPUTS and
TARGETS. The weights are updated every BATCH-SIZE inputs."
  (let ((learning-rate (coerce learning-rate 'double-float)))
    (do ((inputs inputs (rest inputs))
         (targets targets (rest targets))
         (n 0 (1+ n)))
        (nil)
      (declare (type fixnum n))
      (when (or (endp inputs) (= n batch-size))
        (when (>= n 2)
          (average-gradients neural-network n))
        (update-weights-and-biases neural-network learning-rate)
        (setf n 0)
        (when (endp inputs)
          (return)))
      (set-input neural-network (first inputs))
      (propagate neural-network)
      (compute-output-delta neural-network (first targets))
      (backpropagate neural-network))
    (values)))

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
  (let ((layer-sizes (mapcar #'length (neural-network-layers neural-network))))
    (cl-store:store (list layer-sizes
                          (neural-network-weights neural-network)
                          (neural-network-biases neural-network))
                    place)
    (values)))

(defun restore (place)
  "Restore the neural network stored in PLACE, which must be a stream or
a pathname-designator."
  (destructuring-bind (layer-sizes weights biases) (cl-store:restore place)
    (let ((layers (mapcar #'make-double-float-array layer-sizes))
          (gradients (mapcar (lambda (weights)
                               (make-double-float-array (length weights)))
                             weights))
          (deltas (mapcar #'make-double-float-array (rest layer-sizes))))
      (make-neural-network :layers layers
                           :weights weights
                           :biases biases
                           :gradients gradients
                           :deltas deltas))))

(defun copy (neural-network)
  "Return a copy of the NEURAL-NETWORK."
  (let ((layers (neural-network-layers neural-network))
        (weights (neural-network-weights neural-network))
        (biases (neural-network-biases neural-network))
        (gradients (neural-network-gradients neural-network))
        (deltas (neural-network-deltas neural-network)))
    (make-neural-network :layers (mapcar #'copy-seq layers)
                         :weights (mapcar #'copy-seq weights)
                         :biases (mapcar #'copy-seq biases)
                         :gradients (mapcar #'copy-seq gradients)
                         :deltas (mapcar #'copy-seq deltas))))

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
  (let ((output (copy-seq (get-output neural-network))))
    (do ((inputs inputs (rest inputs))
         (targets targets (rest targets))
         (good-guesses 0)
         (n 0 (1+ n)))
        ((endp inputs) (/ good-guesses n))
      (predict neural-network (first inputs) output)
      (when (funcall test output (first targets))
        (incf good-guesses)))))

(defun mean-absolute-error (neural-network inputs targets)
  "Return the mean absolute error on the outputs computed by the NEURAL-NETWORK
when testing it with some INPUTS and TARGETS."
  (let* ((output (copy-seq (get-output neural-network)))
         (output-size (length output))
         (output-error (copy-seq output)))
    (do ((inputs inputs (rest inputs))
         (targets targets (rest targets))
         (total-error 0)
         (n 0 (1+ n)))
        ((endp inputs) (/ total-error (* n output-size)))
      (predict neural-network (first inputs) output)
      (map-into output-error #'- output (first targets))
      (incf total-error (reduce #'+ output-error :key #'abs)))))

(defun means (inputs)
  "Return the means of the variables of the INPUTS."
  (let* ((n (length inputs))
         (input-size (length (first inputs)))
         (results (make-double-float-array input-size)))
    (assert (plusp n))
    (dolist (input inputs)
      (dotimes (i input-size)
        (incf (aref results i) (aref input i))))
    (dotimes (i input-size results)
      (setf (aref results i) (/ (aref results i) n)))))

(defun standard-deviations (inputs means)
  "Return the standard deviations of the variables of the INPUTS."
  (let* ((n (length inputs))
         (input-size (length means))
         (results (make-double-float-array input-size)))
    (assert (>= n 2))
    (dolist (input inputs)
      (dotimes (i input-size)
        (let ((x (- (aref input i) (aref means i))))
          (incf (aref results i) (* x x)))))
    (dotimes (i input-size results)
      (setf (aref results i) (sqrt (/ (aref results i) (1- n)))))))

(defun normalize (input means standard-deviations)
  "Return a normalized variant of the INPUT."
  (map 'double-float-array
       (lambda (x m a)
         (if (zerop a) (- x m) (/ (- x m) a)))
       input
       means
       standard-deviations))

(defun denormalize (input means standard-deviations)
  "Return the original input computed from its normalized variant."
  (map 'double-float-array
       (lambda (x m a)
         (if (zerop a) (+ x m) (+ (* x a) m)))
       input
       means
       standard-deviations))

(defun find-normalization (inputs)
  "Return two values. The first is a normalization function taking a input and
returning a normalized input. Applying this normalization function to the
inputs gives a data set in which each variable has mean 0 and standard
deviation 1. The second is a denormalization function that can compute the
original input from the normalized one."
  (let* ((means (means inputs))
         (standard-deviations (standard-deviations inputs means))
         (normalize (lambda (input)
                      (normalize input means standard-deviations)))
         (denormalize (lambda (input)
                        (denormalize input means standard-deviations))))
    (values normalize denormalize)))

(defun find-learning-rate (neural-network inputs targets
                           &key (batch-size 1) (epochs 1) (iterations 10))
  "Return the best learing rate found in ITERATIONS steps of dichotomic search
(between 0 and 1). In each step, the NEURAL-NETWORK is trained EPOCHS times
using some INPUTS, TARGETS and BATCH-SIZE."
  (labels ((cost (learning-rate)
             (let ((nn (copy neural-network)))
               (dotimes (i epochs)
                 (train nn inputs targets learning-rate batch-size))
               (mean-absolute-error nn inputs targets)))
           (middle (x y)
             (/ (+ x y) 2.0d0))
           (stp (i low a m b high cost-a cost-b best)
             (cond
               ((zerop i)
                best)
               ((> cost-a cost-b)
                (let ((d (middle b high)))
                  (stp (1- i) m b (middle m high) d high cost-b (cost d) b)))
               (t
                (let ((d (middle low a)))
                  (stp (1- i) low d (middle low m) a m (cost d) cost-a a))))))
    (stp iterations 0 1/3 1/2 2/3 1 (cost 1/3) (cost 2/3) nil)))
