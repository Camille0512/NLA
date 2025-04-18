properties([
//     pipelineTriggers([githubPush()])
    pipelineTriggers([githubPullRequests()])
])

// // Not figured out yet
// properties([
//     pipelineTriggers([
//         [
//             $class: 'GitHubPRTrigger',
//             events: [
//                 [$class: 'GitHubPROpenEvent']
//             ]
//         ]
//     ])
// ])

pipeline {
    agent any
    tools {
        git 'Default'  // Uses the Git tool configured in Jenkins Global Tools
    }
    stages {
        stage('Checkout PR') {
            steps {
                sh 'echo "Start PR checkout"'
                checkout scm
//                 checkout([
//                     $class: 'GitSCM',
//                     branches: [[name: 'refs/heads/develop']], // ${sha1}
//                     extensions: [
//                         [$class: 'CloneOption', depth: 1, shallow: true]
//                     ],
//                     userRemoteConfigs: [[
//                         url: 'https://github.com/Camille0512/NLA.git',
//                         credentialsId: 'jenkins_nla',
//                         refspec: '+refs/pull/*:refs/remotes/origin/pr/*'
//                     ]]
//                 ])
                sh 'echo "Finish PR checkout"'
            }
        }

        stage('Build & Test') {
            steps {
                sh 'source /Users/camilleli/Programs/venv/bin/activate'
                sh 'pytest lu_decomposition_test.py'
                junit '**/JenkinsLogs/surefire-reports/*.xml'
                sh 'echo "Finish Build & Test"'
            }
        }
    }
}

post {
    always {
        cleanWs()  // Clean workspace after build
    }
}