import javafx.application.Platform
import static qupath.lib.scripting.QP.*

// 1. Define your 10 classes with your colorblind-safe HEX palette
def classMap = [
    "Primordial"  : Integer.parseInt("009E73", 16), // Bluish Green
    "Activated"   : Integer.parseInt("E69F00", 16), // Warm Orange
    "Primary 3a"  : Integer.parseInt("56B4E9", 16), // Sky Blue
    "Primary 3b"  : Integer.parseInt("0072B2", 16), // Deep Blue
    "Secondary"   : Integer.parseInt("CC79A7", 16), // Reddish Purple
    "Antral"      : Integer.parseInt("D55E00", 16), // Vermillion
    "No Nucleus*" : Integer.parseInt("777777", 16), // Medium Gray
    "MOF"         : Integer.parseInt("F0E442", 16), // Bright Yellow
    "Ovary"       : Integer.parseInt("999999", 16)  // Slate Gray
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