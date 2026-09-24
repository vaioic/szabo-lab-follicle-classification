import javafx.application.Platform
import static qupath.lib.scripting.QP.*

// 1. Define your 10 classes with your colorblind-safe HEX palette
def classMap = [
    "Type 1"  : Integer.parseInt("009E73", 16), // Bluish Green
    "Type 2"  : Integer.parseInt("E69F00", 16), // Orange
    "Type 3a" : Integer.parseInt("56B4E9", 16), // Sky Blue
    "Type 3b" : Integer.parseInt("0072B2", 16), // Blue
    "Type 4"  : Integer.parseInt("D55E00", 16), // Vermillion
    "Type 5a" : Integer.parseInt("CC79A7", 16), // Reddish Purple
    "Type 5b" : Integer.parseInt("332288", 16), // Dark Purple
    "Type 6"  : Integer.parseInt("117733", 16), // Dark Green
    "Type 7"  : Integer.parseInt("CC6677", 16), // Rose
    "Type 8"  : Integer.parseInt("88CCEE", 16),  // Pale Cyan
    "No Nucleus" : Integer.parseInt("F2F2F2", 16),
    "MOF" : Integer.parseInt("FFFF00", 16),
    "Ovary" : Integer.parseInt("00FF00", 16)
]

Platform.runLater {
    // 2. Prepare the list of PathClass objects beginning with the mandatory null/None class
    def classList = [getPathClass(null, makeRGB(187, 187, 187))]
    
    // Add your map entries into the initialization list
    classMap.each { name, color ->
        classList.add(getPathClass(name, color))
    }
    
    // 3. Overwrite/Set available path classes in QuPath's UI container
    getQuPath().getAvailablePathClasses().setAll(classList)
    
    // 4. Force override color assignments if the classes already existed in memory
    getPathClass('None').setColor(makeRGB(187, 187, 187))
    classMap.each { name, color ->
        getPathClass(name).setColor(color)
    }
    
    // 5. Force viewer redraw for any open images
    fireHierarchyUpdate()
    println("Successfully updated all classes and colors dynamically via UI thread!")
}