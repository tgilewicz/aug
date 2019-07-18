import aug


class Operation(aug.Pipeline):

    def apply_on_image(self, image):
        """Apply transformations on image.
            An image is not modified by default. If operation requires an image modification
             then this method should be overwritten.
        """
        return image

    def apply(self, sample):
        image = self.apply_on_image(sample.image)
        annotation = self.apply_on_annotations(
            sample.annotations) if sample.annotations is not None else None

        masks = self.apply_on_masks(sample.masks) if sample.masks is not None else None

        return aug.Sample(image, annotation, masks)

    def apply_on_annotations(self, annotations):
        """Apply transformations on annotations.
            Annotations are not modified by default. If operation requires a modification
                of annotations then this method should be overwritten.
        """
        return annotations

    def apply_on_masks(self, masks):
        """Apply transformations on mask.
            A mask is not modified by default. If operation requires a mask modification
             then this method should be overwritten.
        """
        return masks
