import qupath.lib.objects.PathAnnotationObject
import qupath.lib.common.GeneralTools

// Get current image data
def imageData = getCurrentImageData()
if (imageData == null) {
    println("No image open!")
    return
}

// 1. Create and verify the "export" folder inside the project base directory
def exportDirPath = buildFilePath(PROJECT_BASE_DIR, "export")
def exportDir = new File(exportDirPath)
if (!exportDir.exists()) {
    exportDir.mkdirs()
}

// 2. Extract the image name and strip its file extension
def rawName = imageData.getServer().getMetadata().getName()
def imageName = GeneralTools.stripExtension(rawName)

// 3. Define the final output path inside the export folder
def outputPath = buildFilePath(exportDirPath, "${imageName}_bounding_boxes.csv")
def file = new File(outputPath)

// Write CSV header
file.text = "Class,XMin,YMin,XMax,YMax\n"

// Loop through all annotations and extract info
getAnnotationObjects().each { ann ->
    def roi = ann.getROI()
    if (roi != null) {
        def classification = ann.getPathClass() ? ann.getPathClass().toString() : "Unclassified"
        def xmin = roi.getBoundsX()
        def ymin = roi.getBoundsY()
        def xmax = xmin + roi.getBoundsWidth()
        def ymax = ymin + roi.getBoundsHeight()
        
        // Append to CSV file
        file.append("\"${classification}\",${xmin},${ymin},${xmax},${ymax}\n")
    }
}

println("Exported bounding boxes to: " + outputPath)